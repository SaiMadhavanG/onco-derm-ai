# Initialization

- [x] come up with a name
- [x] kedro init
- [x] pre commit hooks
  - [x] linting
  - [x] testing
  - [x] quartodoc
- [x] testing - _pytest, git pre commit hook_
- [x] auto documentation - _quartodoc_

# Documentation

- [x] project card - **sai**
- [x] data card
- [ ] model card
- [ ] mlops card

# Data Prep

- [x] Load data - _kedro_
- [x] Data versioning - _kedro_
- [x] Image preprocessing - normalizing, tensorizing and resizing - _torchvision_ - **srini**
- [ ] Data augmentation - _torchvision_
  - [ ] Data quality - **?**, _greater expectation_
  - [ ] right to erasure, forgetting - _kedro pipeline_

# Training

- [x] Model training - Resnet - _PyTorch_
- [x] Model eval - _Sk classification report_
- [x] Model versioning - _mlflow_
- [ ] hyperparameter tuning - _optuna/sklearn search_
- [ ] automatic reports - _quarto and plotting libs_
- [ ] Model pruning - _pytorch_
- [ ] adversarial robustness - _auto_lirpa_
- [ ] right to erasure, model retraining - _kedro pipeline_

# Inference

- [ ] explainability - _deel_
- [ ] confidence calibration - _deel_, **?**
- [ ] OOD detection - **?**
- [ ] Conformal predictions - _deel_

# Deployment

- [ ] model containerization - _mlflow, docker_
- [ ] auto model deployment - _github actions/cloud provider_
- [ ] deployment side eval
- [ ] data drift
- [ ] auto retraining triggers

# LLM

- [ ] llm set up
- [ ] llm prompt config
- [ ] llm deployment

# Front End

- [ ] Frontend dashboard - _react, bootstrap_
- [ ] Chatbot window
