"""Preprocessing and machine-learning pipeline construction."""

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder


NOMINAL_COLUMNS = [
    "policy_state",
    "policy_csl",
    "insured_sex",
    "insured_occupation",
    "insured_hobbies",
    "incident_type",
    "collision_type",
    "authorities_contacted",
    "incident_state",
    "incident_city",
    "property_damage",
    "police_report_available",
    "auto_model",
]

ORDINAL_COLUMNS = [
    "insured_education_level",
    "incident_severity",
]

NUMERICAL_COLUMNS = [
    "months_as_customer",
    "age",
    "policy_deductable",
    "policy_annual_premium",
    "umbrella_limit",
    "capital-gains",
    "capital-loss",
    "incident_hour_of_the_day",
    "number_of_vehicles_involved",
    "bodily_injuries",
    "witnesses",
    "total_claim_amount",
    "injury_claim",
    "property_claim",
    "vehicle_claim",
    "Duration",
    "auto_year",
]


def build_preprocessor() -> ColumnTransformer:
    """Build preprocessing for numerical, nominal, and ordinal variables."""

    education_order = [
        "Associate",
        "College",
        "High School",
        "Masters",
        "JD",
        "MD",
        "PhD",
    ]

    severity_order = [
        "Trivial Damage",
        "Minor Damage",
        "Major Damage",
        "Total Loss",
    ]

    numerical_transformer = Pipeline(
        steps=[("scaler", MinMaxScaler())]
    )

    nominal_transformer = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            )
        ]
    )

    ordinal_transformer = Pipeline(
        steps=[
            (
                "ordinal",
                OrdinalEncoder(
                    categories=[education_order, severity_order],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
            ("scaler", MinMaxScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numerical", numerical_transformer, NUMERICAL_COLUMNS),
            ("nominal", nominal_transformer, NOMINAL_COLUMNS),
            ("ordinal", ordinal_transformer, ORDINAL_COLUMNS),
        ],
        remainder="drop",
    )


def build_model_pipeline(
    classifier,
    random_state: int = 42,
) -> ImbPipeline:
    """Build a preprocessing + SMOTE + classifier pipeline."""
    return ImbPipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("smote", SMOTE(random_state=random_state)),
            ("classifier", classifier),
        ]
    )
