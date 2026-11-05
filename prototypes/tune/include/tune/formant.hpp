// Formant preservation for the Track-mode shifter (spec §6.5).
//
//   x --whiten A(D(z)), a(t)--> e --shifter--> e' --recolour 1/A(D(z)), a(t_in)--> y
//
// The whitening filter removes the vocal-tract envelope, so resampling in the
// shifter moves only the excitation (pitch), not the formants. The recolour
// filter puts the envelope back using the coefficients of the input moment
// the shifter is reading (t_in), so the envelope stays aligned through the
// shifter's delay wander.
//
// Frequency-warped LPC (spike v0.2): every unit delay is replaced by the
// first-order all-pass D(z) = (z^-1 - lambda) / (1 - lambda z^-1). With
// lambda ~ 0.75 at 48 kHz (Bark-like) the model's resolution concentrates
// below ~4 kHz, where formants are; plain LPC (lambda = 0) at 48 kHz spreads
// its poles up to 24 kHz and under-models F1/F2 by ~8 dB at order 16
// (measured, spike report).
//
// The warped all-pole recolour filter has a delay-free loop (D(z) has a
// direct term -lambda). It is solved per sample in closed form: every chain
// value is v_k = (-lambda)^k * y + s_k, where s_k depends only on past state,
// so y = (e' - sum a_k s_k) / (sum a_k (-lambda)^k). With identical
// coefficient sequences, whitening and recolouring are exact inverses even
// while the coefficients change: at ratio 1 the chain is a pure delay.
//
// Coefficients: warped autocorrelation of a (optionally pre-emphasised) 20 ms
// Hann frame every 5 ms hop, Levinson -> reflection coefficients (interpolated
// between frames, which keeps them stable) -> direct-form coefficients per
// 16-sample block. Trajectory for hop segment s interpolates frame s-2 -> s-1,
// both already computed when segment s starts (causal).
//
// Real-time safe: allocation only in prepare().
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "tune/lpc.hpp"

namespace tune {

struct FormantConfig {
    double window_ms = 20.0;
    double hop_ms = 10.0;          // standard speech-coding rate; interpolated per 16-sample block
    int order = 20;               // warped order (not scaled with fs: warping handles it)
    double lambda = -1.0;         // all-pass warping; < 0 = Bark-like default for the sample rate
    double pre_emphasis = 0.0;    // analysis only
    // Defaults chosen in spike v0.2 (sweep in the spike report): the noise
    // floor limits how far whitening boosts near-silent bands, and the lag
    // window smooths the envelope so high voices' harmonics are not modelled
    // as formants (which delayed onsets at ~300 Hz).
    double white_noise = 3e-2;    // noise floor added to r[0] (relative)
    double lag_window_hz = 100.0; // Gaussian lag window on the warped autocorrelation; 0 = off
    int block = 16;               // samples per direct-form coefficient update
    int history_frames = 32;      // must cover the shifter's maximum delay
};

// Smith & Abel's Bark-warping all-pass coefficient (0.756 at 48 kHz).
inline double bark_lambda(double fs) {
    return 1.0674 * std::sqrt(2.0 / 3.14159265358979323846 * std::atan(0.06583 * fs / 1000.0)) - 0.1916;
}

class FormantChain {
public:
    void prepare(double fs, const FormantConfig& cfg = {}) {
        fs_ = fs;
        cfg_ = cfg;
        order_ = std::clamp(cfg.order, 4, 64);
        lambda_ = cfg.lambda < 0.0 ? bark_lambda(fs) : std::clamp(cfg.lambda, 0.0, 0.95);
        win_len_ = std::max(64, static_cast<int>(std::lround(cfg.window_ms * 1e-3 * fs)));
        hop_ = std::max(16, static_cast<int>(std::lround(cfg.hop_ms * 1e-3 * fs)));
        block_ = std::max(1, cfg.block);
        std::size_t size = 1;
        while (size < static_cast<std::size_t>(win_len_)) size <<= 1;
        ana_.assign(size, 0.0);
        ana_mask_ = size - 1;
        window_.resize(win_len_);
        for (int i = 0; i < win_len_; ++i)
            window_[i] = 0.5 - 0.5 * std::cos(2.0 * 3.14159265358979323846 * (i + 0.5) / win_len_);
        frame_.assign(win_len_, 0.0);
        chain_.assign(win_len_, 0.0);
        chain2_.assign(win_len_, 0.0);
        r_.assign(order_ + 1, 0.0);
        a_.assign(order_ + 1, 0.0);
        tmp_.assign(order_ + 1, 0.0);
        frames_.assign(static_cast<std::size_t>(cfg.history_frames) * order_, 0.0);
        kbuf_.assign(order_, 0.0);
        aw_.assign(order_ + 1, 0.0);
        ar_.assign(order_ + 1, 0.0);
        gpow_.assign(order_ + 1, 1.0);
        for (int k = 1; k <= order_; ++k) gpow_[k] = gpow_[k - 1] * (-lambda_);
        gpow_r_ = gpow_;
        lambda_r_ = lambda_;
        scale_r_ = 1.0;
        vw_.assign(order_ + 1, 0.0);
        vr_.assign(order_ + 1, 0.0);
        sr_.assign(order_ + 1, 0.0);
        reset();
    }

    void reset() {
        std::fill(ana_.begin(), ana_.end(), 0.0);
        std::fill(frames_.begin(), frames_.end(), 0.0);
        std::fill(vw_.begin(), vw_.end(), 0.0);
        std::fill(vr_.begin(), vr_.end(), 0.0);
        std::fill(aw_.begin(), aw_.end(), 0.0);
        std::fill(ar_.begin(), ar_.end(), 0.0);
        aw_[0] = ar_[0] = 1.0;
        aw_block_ = ar_block_ = -1;
        lambda_r_ = lambda_;
        scale_r_ = 1.0;
        for (int k = 0; k <= order_; ++k) gpow_r_[k] = gpow_[k];
        n_ = 0;
        prev_x_ = 0.0;
        frames_done_ = 0;
    }

    int order() const { return order_; }
    int hop() const { return hop_; }
    double lambda() const { return lambda_; }
    double recolour_lambda() const { return lambda_r_; }

    // Throat (spec §6.5): re-colour with a different warping so the whole
    // envelope moves by `scale` along the frequency axis (> 1 = formants up,
    // a smaller / brighter voice). The re-colour warping is solved in closed
    // form so that a 1 kHz reference lands exactly at scale x 1 kHz:
    //   Omega_lw(w_ref) = Omega_lr(scale * w_ref),
    //   Omega_l(w) = w + 2 atan(l sin w / (1 - l cos w)).
    // scale 1 restores lambda_r = lambda (exact inverse). Real-time safe.
    void set_recolour_scale(double scale) {
        if (std::abs(scale - scale_r_) < 1e-7) return;
        scale_r_ = scale;
        if (std::abs(scale - 1.0) < 1e-9) {
            lambda_r_ = lambda_;
        } else {
            const double pi = 3.14159265358979323846;
            const double w_ref = 2.0 * pi * 1000.0 / fs_;
            const double big_omega =
                w_ref + 2.0 * std::atan(lambda_ * std::sin(w_ref) / (1.0 - lambda_ * std::cos(w_ref)));
            const double w2 = std::min(scale * w_ref, 0.9 * pi);
            const double t = std::tan(0.5 * (big_omega - w2));
            lambda_r_ = std::clamp(t / (std::sin(w2) + t * std::cos(w2)), -0.95, 0.95);
        }
        for (int k = 1; k <= order_; ++k) gpow_r_[k] = gpow_r_[k - 1] * (-lambda_r_);
    }

    // Whiten the input sample at input time n_ (incremented here).
    float whiten(float xin) {
        const double x = xin;
        ana_[static_cast<std::size_t>(n_) & ana_mask_] = x - cfg_.pre_emphasis * prev_x_;
        prev_x_ = x;

        const std::int64_t b = n_ / block_;
        if (b != aw_block_) {
            direct_coefficients_at(static_cast<double>(b * block_), aw_.data());
            aw_block_ = b;
        }
        // Warped FIR: v_0 = x, v_k = -lambda v_{k-1} + v_{k-1}[n-1] + lambda v_k[n-1].
        double prev_km1 = vw_[0];  // v_{k-1}[n-1]
        vw_[0] = x;
        double e = x;
        for (int k = 1; k <= order_; ++k) {
            const double old_k = vw_[k];
            const double vk = -lambda_ * vw_[k - 1] + prev_km1 + lambda_ * old_k;
            prev_km1 = old_k;
            vw_[k] = vk;
            e += aw_[k] * vk;
        }
        if ((n_ + 1) % hop_ == 0) analyse(n_ / hop_);
        ++n_;
        return static_cast<float>(e);
    }

    // Recolour one shifter output sample whose content comes from input time
    // `t_in` (samples; floor is used).
    float recolour(float ein, double t_in) {
        const double t = std::floor(t_in);
        const std::int64_t b = t < 0.0 ? -1 : static_cast<std::int64_t>(t) / block_;
        if (b != ar_block_) {
            direct_coefficients_at(b < 0 ? -1.0 : static_cast<double>(b * block_), ar_.data());
            ar_block_ = b;
        }
        // s_0 = 0; s_k = -lambda s_{k-1} + v_{k-1}[n-1] + lambda v_k[n-1].
        double num = ein, den = 1.0;
        sr_[0] = 0.0;
        for (int k = 1; k <= order_; ++k) {
            sr_[k] = -lambda_r_ * sr_[k - 1] + vr_[k - 1] + lambda_r_ * vr_[k];
            num -= ar_[k] * sr_[k];
            den += ar_[k] * gpow_r_[k];
        }
        const double y = num / den;
        for (int k = 0; k <= order_; ++k) vr_[k] = gpow_r_[k] * y + sr_[k];
        return static_cast<float>(y);
    }

    // Reflection coefficients of the trajectory at input time t (k[0..order)).
    void reflection_at(double t, double* k) const {
        if (t < 0.0) {
            for (int i = 0; i < order_; ++i) k[i] = 0.0;
            return;
        }
        const std::int64_t ti = static_cast<std::int64_t>(t);
        const std::int64_t s = ti / hop_;
        const double alpha = static_cast<double>(ti - s * hop_) / hop_;
        const double* f1 = frame_ptr(s - 1);
        const double* f0 = frame_ptr(s - 2);
        for (int i = 0; i < order_; ++i) {
            const double a = f0 ? f0[i] : 0.0;
            const double c = f1 ? f1[i] : 0.0;
            k[i] = a + (c - a) * alpha;
        }
    }

private:
    // Reflection -> direct form (step-up recursion), a[0] = 1.
    void direct_coefficients_at(double t, double* a) {
        reflection_at(t, kbuf_.data());
        a[0] = 1.0;
        for (int i = 1; i <= order_; ++i) a[i] = 0.0;
        for (int i = 1; i <= order_; ++i) {
            const double ki = kbuf_[i - 1];
            for (int j = 1; j < i; ++j) tmp_[j] = a[j] + ki * a[i - j];
            for (int j = 1; j < i; ++j) a[j] = tmp_[j];
            a[i] = ki;
        }
    }

    const double* frame_ptr(std::int64_t j) const {
        if (j < 0 || j >= frames_done_ || frames_done_ - j > cfg_.history_frames) return nullptr;
        return &frames_[static_cast<std::size_t>(j % cfg_.history_frames) * order_];
    }

    void analyse(std::int64_t j) {
        const std::int64_t end = n_;  // newest sample, inclusive
        for (int i = 0; i < win_len_; ++i) {
            const std::int64_t idx = end - (win_len_ - 1) + i;
            frame_[i] = idx >= 0 ? ana_[static_cast<std::size_t>(idx) & ana_mask_] * window_[i] : 0.0;
        }
        // Warped autocorrelation: r[m] = sum_i x[i] * (D^m x)[i].
        double r0 = 0.0;
        for (int i = 0; i < win_len_; ++i) r0 += frame_[i] * frame_[i];
        r_[0] = r0;
        std::copy(frame_.begin(), frame_.end(), chain_.begin());
        for (int m = 1; m <= order_; ++m) {
            // chain2 = D(chain), zero initial state.
            double x1 = 0.0, y1 = 0.0;
            for (int i = 0; i < win_len_; ++i) {
                const double y = -lambda_ * chain_[i] + x1 + lambda_ * y1;
                x1 = chain_[i];
                y1 = y;
                chain2_[i] = y;
            }
            double acc = 0.0;
            for (int i = 0; i < win_len_; ++i) acc += frame_[i] * chain2_[i];
            r_[m] = acc;
            chain_.swap(chain2_);
        }
        if (cfg_.lag_window_hz > 0.0) {
            const double pi = 3.14159265358979323846;
            for (int m = 1; m <= order_; ++m) {
                const double x = 2.0 * pi * cfg_.lag_window_hz * m / fs_;
                r_[m] *= std::exp(-0.5 * x * x);
            }
        }
        r_[0] *= 1.0 + cfg_.white_noise;
        double* out = &frames_[static_cast<std::size_t>(j % cfg_.history_frames) * order_];
        levinson(r_.data(), order_, out, a_.data(), tmp_.data());
        frames_done_ = j + 1;
    }

    double fs_ = 48000.0;
    FormantConfig cfg_{};
    int order_ = 20, win_len_ = 960, hop_ = 240, block_ = 16;
    double lambda_ = 0.75;
    double lambda_r_ = 0.75, scale_r_ = 1.0;  // re-colour warping (Throat)
    std::vector<double> ana_;
    std::size_t ana_mask_ = 0;
    std::vector<double> window_, frame_, chain_, chain2_, r_, a_, tmp_;
    std::vector<double> frames_;  // history_frames x order (reflection coefficients)
    std::vector<double> kbuf_, aw_, ar_, gpow_, gpow_r_, vw_, vr_, sr_;
    std::int64_t aw_block_ = -1, ar_block_ = -1;
    std::int64_t n_ = 0, frames_done_ = 0;
    double prev_x_ = 0.0;
};

}  // namespace tune
