# Machine Learning Interactive Visualization

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/dhaitz/machine-learning-interactive-visualization/master?urlpath=voila%2Frender%2Fmachine-learning-interactive-visualization.ipynb)

An interactive dashboard made with Jupyter and [Voila](https://github.com/QuantStack/voila).
Users can play around with parameter like class imbalance, model strength or cutoff value and observe the effects on metris like ROC/AUC or accuracy/precision/recall.

Now available live [here](https://machine-learning-visualization.herokuapp.com/) and in the [official Voila Gallery](https://voila-gallery.org/services/gallery/).

Some code copied from [this voila example](https://github.com/pbugnion/voila-gallery/blob/master/country-indicators/index.ipynb).

Deployment: https://voila.readthedocs.io/en/latest/deploy.html#deployment-on-heroku

![ml](img/ml_visualization.gif)


## ToDo

- More comprehensive Readme on machine learning model evaluation, describing metrics in detail
- Create GIFs and add to Readme
- Smoother refresh and faster plot update. Could be possible via using matplotlib's notebook or widget backend, but currently not working, see [this GitHub issue](https://github.com/matplotlib/matplotlib/issues/15076).
