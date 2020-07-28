# Test docker deploy of model

Inspired by these blog post: 
- https://winderresearch.com/a-simple-docker-based-workflow-for-deploying-a-machine-learning-model/

- https://nickc1.github.io/api,/scikit-learn/2019/01/10/scikit-fastapi.html

- https://www.datacamp.com/community/tutorials/machine-learning-models-api-python


## Future reading

- About using Pydantic better, https://engineering.upside.com/building-better-machine-learning-feature-validation-using-pydantic-2fc99990faf0

## Possible Todos

- get the main app to work inside a docker image
- Accept multiple inputs (ie. 10 rows of features)
- Include the preprocessing step in the modelling pipeline, and export it
- Load and use it in the main.py
- Warning / special response with missign fields.