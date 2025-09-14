## Model Interpretation Summary (SHAP Analysis)

The SHAP analysis indicates that the model’s predictions are primarily driven by four key factors, in order of importance:

1. **Property size (`squareMeters`)**  
2. **Location (`city`)**  
3. **Construction year (`buildYear`)**  
4. **Proximity to the city center (`centreDistance`)**

In addition, property prices exhibit **seasonal fluctuations across months**.

### Secondary Features
Secondary features further refine the estimates, with the most relevant being:

- **Proximity to points of interest**: schools, clinics, restaurants  
- **Amenities**: elevators, balconies  

While these variables have smaller individual effects, they enhance the model by incorporating **practical and contextual influences** on property values.
