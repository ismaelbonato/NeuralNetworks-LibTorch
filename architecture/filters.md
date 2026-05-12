# Filters linear vs nonlinear

### Mostly Linear
```
Low-pass: removes/reduces highs; does not create new frequencies.
High-pass: removes/reduces lows.
Band-pass: keeps a middle band.
Band-stop/notch: removes a narrow band.
Graphic EQ: boosts/cuts fixed frequency bands.
Parametric EQ: boosts/cuts selected bands.
Shelving EQ: boosts/cuts lows or highs.
FIR cabinet impulse response: convolution with a fixed impulse response.
Simple convolution reverb: fixed impulse response, same behavior regardless of level.
Delay/echo with fixed feedback if no saturation/modulation: delayed copies are scaled and summed.
Phase shifter with fixed coefficients and no modulation: fixed phase changes.
Linear-phase EQ: frequency shaping with phase behavior controlled.
All-pass filter with fixed coefficients: changes phase, not magnitude.
Moving average filter: averages nearby samples.
De-esser as fixed EQ: linear only if it is not level-triggered.
```

### Nonlinear / Dynamic

```
Distortion: clips/warps the waveform, creating harmonics.
Overdrive: soft clipping, level-dependent harmonic generation.
Fuzz: strong nonlinear clipping/gating.
Saturation: level-dependent coloration and harmonics.
Compressor: gain changes based on signal level.
Limiter: strong compression near a threshold.
Expander: gain changes based on level.
Noise gate: opens/closes based on level.
Envelope filter/auto-wah: filter changes based on input envelope.
Dynamic EQ: EQ changes based on level.
De-esser as a processor: reduces sibilance only when detected.
Amp model: usually nonlinear saturation plus tone filtering.
Speaker/cab model with cone breakup: nonlinear if response changes with level.
Chorus: time-varying delay/modulation; not time-invariant, often not simple linear for learning.
Flanger: modulated delay/comb filtering.
Phaser with LFO: time-varying all-pass stages.
Tremolo: amplitude modulation over time.
Ring modulator: multiplies signals, creates sidebands.
Pitch shifter: changes frequency/time relationship.
Octaver: often nonlinear pitch detection/synthesis.
Bitcrusher: quantization and sample-rate reduction create artifacts.
Wavefolder: folds waveform, creating harmonics.
Rectifier: flips/clips signal portions, creates harmonics.
Wah pedal: filter may be linear at a fixed pedal position, but time-varying when moved.
```