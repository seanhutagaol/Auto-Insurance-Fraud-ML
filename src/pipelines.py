"""Pipelines Module.

This module constructs preprocessing transformers (scaling, one-hot, 
ordinal mapping) and wraps them inside imbalanced-aware machine learning pipelines.
"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


def build_preprocessor() -> ColumnTransformer:
    """Construct a ColumnTransformer for numerical, nominal, and ordinal features.

    Returns
    -------
    ColumnTransformer
        The configured scikit-learn preprocessor object.
    """
    nominal_columns = [
        'policy_state', 'policy_csl', 'insured_sex', 'insured_occupation', 
        'insured_hobbies', 'incident_type', 'collision_type', 'authorities_contacted', 
        'incident_state', 'incident_city', 'property_damage', 'police_report_available', 
        'auto_model'
    ]
    
    ordinal_columns = ['insured_education_level', 'incident_severity']
    
    numerical_columns = [
        'months_as_customer', 'age', 'policy_deductable', 'policy_annual_premium', 
        'umbrella_limit', 'capital-gains', 'capital-loss', 'incident_hour_of_the_day', 
        'number_of_vehicles_involved', 'bodily_injuries', 'witnesses', 
        'total_claim_amount', 'injury_claim', 'property_claim', 'vehicle_claim', 
        'Duration', 'auto_year'
    ]

    # Pre-defined logical orders for ordinal features
    iel_order = ['Associate', 'College', 'High School', 'Masters', 'JD', 'MD', 'PhD']
    is_order = ['Trivial Damage', 'Minor Damage', 'Major Damage', 'Total Loss']

    numerical_transformer = Pipeline(steps=[('Scaler', MinMaxScaler())])
    
    nominal_transformer = Pipeline(steps=[
        ('Onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    ordinal_transformer = Pipeline(steps=[
        ('Ordinal', OrdinalEncoder(categories=[iel_order, is_order])),
        ('Scaler', MinMaxScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('Numerical', numerical_transformer, numerical_columns),
            ('Nominal', nominal_transformer, nominal_columns),
            ('Ordinal', ordinal_transformer, ordinal_columns)
        ], 
        remainder='passthrough'
    )
    
    return preprocessor


def build_model_pipeline(classifier) -> ImbPipeline:
    """Wrap a classifier inside an imbalanced-aware pipeline containing SMOTE and preprocessing.

    Parameters
    ----------
    classifier : estimator object
        An un-fitted scikit-learn compatible classifier (e.g., RandomForestClassifier).

    Returns
    -------
    ImbPipeline
        The complete imbalance-safe modeling pipeline.
    """
    preprocessor = build_preprocessor()
    
    pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', classifier)
    ])
    
    return pipeline
