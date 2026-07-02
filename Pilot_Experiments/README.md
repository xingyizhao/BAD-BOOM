# Pilot Experiments

This directory contains two pilot experiments used in BAD-BOOM:

1. Backdoor forgetting across downstream fine-tuning epochs.
2. Backdoor basin and ASR basin visualization in the local parameter landscape.

## 1. Backdoor Forgetting

This experiment corresponds to **BAD-BOOM Figure 1**. It studies how the backdoor effect changes during downstream fine-tuning by plotting:

- **ASR**: attack success rate
- **ACC**: downstream task accuracy

The script reads the recorded results from `./Data/` and generates the figure for a selected threat scenario and attack model.

<p align="center">
  <img src="./Figure/BAD-BOOM%20Figure%201.jpg" alt="BAD-BOOM Figure 1" width="650">
</p>

### Run the experiment

Example:

```bash
python backdoor_forgetting.py --threat_scenario sentiment_steering --attack_model AddSent
```

Supported arguments:

- `--threat_scenario`: `sentiment_steering` or `targeted_refusal`
- `--attack_model`: `AddSent`, `Sleeper`, or `VPI`

Another example:

```bash
python backdoor_forgetting.py --threat_scenario targeted_refusal --attack_model VPI
```

The output figure is saved to `Figure/<threat_scenario>_<attack_model>.png`.

## 2. Backdoor Basin and ASR Basin

This experiment visualizes the local geometry of a poisoned model in two sampled orthogonal directions of parameter space.

- `backdoor_basin.py` computes the loss basin and ASR basin values.
- `landscape_plot.py` turns the saved results into interactive HTML plots.
- `landscape_plot.sh` runs the full workflow.

The generated plots help show both:

- the **backdoor loss basin**
- the **attack success rate basin**

### Interactive HTML plots

The two Plotly HTML files are:

- `Figure/sentiment_steering_VPI_Qwen3-0.6B_BAD-BOOM.html`
- `Figure/sentiment_steering_VPI_Qwen3-0.6B_AdamW.html`

**Important note.** To view the basin interactively, download the HTML files and open them in a local browser.

### Run the experiment

The easiest way is:

```bash
bash landscape_plot.sh
```

This shell script runs:

1. `backdoor_basin.py` to compute the landscape data.
2. `landscape_plot.py` to export the HTML visualizations.

After the script finishes, open the generated HTML files in your browser to inspect the interactive loss basin and ASR basin.

If you want to run the first stage manually, the command format is:

```bash
python backdoor_basin.py \
  --base_model_name_or_path "Qwen/Qwen3-0.6B-Base" \
  --backdoor_model_name_or_path "../Saved_Models/sentiment_steering/AddSent/Qwen_0.6B/Attacker/AdamW" \
  --threat_scenario "sentiment_steering" \
  --backdoor_attack_method "AddSent" \
  --optimizer "AdamW" \
  --model_series "Qwen3-0.6B" \
  --max_length 512 \
  --sample_size 128 \
  --per_device_batch_size 128 \
  --seed 1001
```

Then generate the interactive plots with:

```bash
python landscape_plot.py \
  --threat_scenario "sentiment_steering" \
  --backdoor_attack_method "AddSent" \
  --model_series "Qwen3-0.6B" \
  --optimizer "AdamW"
```
