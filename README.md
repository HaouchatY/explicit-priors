# Explicit priors

Fields-of-experts potentials computed by linear least squares instead of training.
On WCRR's own 60 filters, MAP with these potentials beats the trained WCRR at β = σ², with no tuning.

| σ = 25 (PSNR, dB) | Set12 | BSD68 |
|---|---|---|
| WCRR (trained) | 29.618 | 28.681 |
| unfiltered (−log marginals) | 28.985 | 27.873 |
| **filtered, MAP, β = σ²** | **29.742** | **28.712** |
| two closed-form steps, WCRR60 + V4 bank | 29.922 | 28.818 |

Fit: 2.8 min on a V100, against 3 h to train WCRR.

**Idea.** Filter histograms are projections of the image density. Maximum entropy turns them into a fields of experts. Score-matching normal equations `A c = d` act as a filtered back-projection: `d` comes from the histograms, `A⁻¹` undoes the filter overlap. Gribonval's MMSE–prox identity then turns the fitted denoiser into MAP potentials.

```bash
export AFOE_IMG=/path/to/BSD AFOE_OUT=./out    # BSD has train/, Set12/, Set68/
cd afoe
python afoe.py fit  --name W --bank WCRR60 --r 3
python afoe.py fit  --name Wc --chain W --r 3
python afoe.py prox --name Wp --chain W,Wc --bank WCRR60 --r 3
python afoe.py map  --name Wp --variant jmono
```

`afoe/`: code · `data/`: filter banks · `results/`: all numbers · `scripts/`: cluster jobs · `slides/`: talk
