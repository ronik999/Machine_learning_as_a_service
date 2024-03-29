Machine_Learning_as_a_service_assignment_2
==============================

This project is for creating a machine learning model and forecasting model to predict the revenue for the American 
retailer that has 10 stores across 3 different states: California (CA), Texas (TX), and Wisconsin (WI).   
Two different types of problems were identified and solved using the models for this project.   
There are 4 endpoints created for this project:    
- / (GET): First endpoint to reach this page.  
- /health/ (GET): Second endpoint for a welcoming message.   
- /sales/national/ (GET): Third endpoint for getting the prediction of the forecasting model. The model expects a date 
in string format. Format - (yyyy/mm/dd): Where y is year, m is month and d is day. 
- /sales/stores/items/ (GET): Fourth endpoint for the predictive model. The model expects 3 parameters as input: item id, store id, and date for prediction. Format for date is similar to third endpoint

All the required files are attached in this repository for assignment 2.
The FastApi is deployed on heroku.

Link to Heroku: https://mla-at2-e0654ce6deb8.herokuapp.com/

Project Organization
------------

    ├── LICENSE
    ├── app            <- Building FastApi files
    │   └── main.py 
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io
    └── DockerFile  <- For building docker image
    └── Heroku.yml  <- For deploying to heroku


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
