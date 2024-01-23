# Scaling-laws-of-transcription
Customized code used to generate results reported in the manuscript "Scaling laws of human transcriptional activity"

#### System Requirements

The code requires only a standard computer with enough RAM to support the in-memory operations. It has been tested on a 2.3 GHz Quad-Core Intel Core i7 MacBook Pro with 16 GB memory and operating system macOS Sonoma 14.2.1.

##### Python Dependencies

The code mainly depends on the Python scientific stack.

```
jupyter==1.0.0
numpy==1.23.5
pands==2.0.2
plotly==5.17.0
scipy=1.10.1
statsmodels=0.14.0
```

Run `pip3 install -r requirements.txt` to install the above dependencies used in the project.

#### Usage guide

`RegressionModel.py`, `scale_plot_super.py`, and `statsRunner.py` are auxiliary functions. 

Run `Scaling_Plots_and_Stats.ipynb` to generate all the plots. The expected output is exactly as shown in the paper. All plots are expected to be generated within minutes.

#### Contact

Questions can be addressed to Jiayin Hong at jh2313#AT#cam.ac.uk.

#### License

This project is covered under the MIT license.
