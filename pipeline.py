
import os
import boto3
import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.conditions import ConditionGreaterThan
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
    ParameterBoolean
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.condition_step import ConditionStep, JsonGet
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.model import Model
from sagemaker.workflow.steps import CreateModelStep
from sagemaker.inputs import CreateModelInput, TransformInput
from sagemaker.workflow.steps import TransformStep
from sagemaker.transformer import Transformer
from sagemaker.tuner import (
    IntegerParameter,
    ContinuousParameter,
    HyperparameterTuner
)
from sagemaker.workflow.steps import TuningStep

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

def get_session(region, default_bucket):
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
    )

def get_pipeline(
    region,
    role=None,
    default_bucket=None,
    model_package_group_name="ChurnModelPackageGroup",
    pipeline_name="ChurnModelPipeline",
    base_prefix=None,
    custom_image_uri=None,
    sklearn_processor_version=None
    ):

    sagemaker_session = get_session(region, default_bucket)
    
    # Parameters
    processing_instance_count = ParameterInteger(name="ProcessingInstanceCount", default_value=1)
    processing_instance_type = ParameterString(name="ProcessingInstanceType", default_value="ml.m5.large")
    training_instance_type = ParameterString(name="TrainingInstanceType", default_value="ml.m5.large")
    input_data = ParameterString(name="InputData", default_value=f"s3://{default_bucket}/data/storedata_total.csv")
    batch_data = ParameterString(name="BatchData", default_value=f"s3://{default_bucket}/data/batch/batch.csv")
    use_spot_instances = ParameterBoolean(name="UseSpotInstances", default_value=True)
    max_run = ParameterInteger(name="MaxRun", default_value=9000)
    max_wait = ParameterInteger(name="MaxWait", default_value=10000)
    
    # Processing Step for Feature Engineering
    sklearn_processor = SKLearnProcessor(
        framework_version=sklearn_processor_version,
        instance_type=processing_instance_type,
        instance_count=processing_instance_count,
        sagemaker_session=sagemaker_session,
        role=role,
    )
    step_process = ProcessingStep(
        name="ChurnModelProcess",
        processor=sklearn_processor,
        inputs=[ProcessingInput(source=input_data, destination="/opt/ml/processing/input")],
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train", destination=f"s3://{default_bucket}/output/train"),
            ProcessingOutput(output_name="validation", source="/opt/ml/processing/validation", destination=f"s3://{default_bucket}/output/validation"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test", destination=f"s3://{default_bucket}/output/test")
        ],
        code=f"s3://{default_bucket}/input/code/preprocess.py",
    )
    
    # Training Step
    model_path = f"s3://{default_bucket}/output"
    image_uri = sagemaker.image_uris.retrieve(
        framework="xgboost",
        region=region,
        version="1.0-1",
        py_version="py3",
        instance_type=training_instance_type,
    )
    fixed_hyperparameters = {
        "eval_metric": "auc",
        "objective": "binary:logistic",
        "num_round": "100",
        "rate_drop": "0.3",
        "tweedie_variance_power": "1.4"
    }
    xgb_train = Estimator(
        image_uri=image_uri,
        instance_type=training_instance_type,
        instance_count=1,
        hyperparameters=fixed_hyperparameters,
        output_path=model_path,
        base_job_name="churn-train",
        sagemaker_session=sagemaker_session,
        role=role,
        use_spot_instances=use_spot_instances,
        max_run=max_run,
        max_wait=max_wait,
    )
    hyperparameter_ranges = {
        "eta": ContinuousParameter(0, 1),
        "min_child_weight": ContinuousParameter(1, 10),
        "alpha": ContinuousParameter(0, 2),
        "max_depth": IntegerParameter(1, 10),
    }
    objective_metric_name = "validation:auc"
    step_tuning = TuningStep(
        name="ChurnHyperParameterTuning",
        tuner=HyperparameterTuner(
            xgb_train,
            objective_metric_name,
            hyperparameter_ranges,
            max_jobs=2,
            max_parallel_jobs=2
        ),
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "validation": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["validation"].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
    )
    
    # Evaluation Step
    script_eval = ScriptProcessor(
        image_uri=image_uri,
        command=["python3"],
        instance_type=processing_instance_type,
        instance_count=1,
        base_job_name="script-churn-eval",
        role=role,
        sagemaker_session=sagemaker_session,
    )
    evaluation_report = PropertyFile(name="ChurnEvaluationReport", output_name="evaluation", path="evaluation.json")
    step_eval = ProcessingStep(
        name="ChurnEvalBestModel",
        processor=script_eval,
        inputs=[
            ProcessingInput(source=step_tuning.get_top_model_s3_uri(top_k=0, s3_bucket=default_bucket, prefix="output"), destination="/opt/ml/processing/model"),
            ProcessingInput(source=step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri, destination="/opt/ml/processing/test")
        ],
        outputs=[ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation", destination=f"s3://{default_bucket}/output/evaluation")],
        code=f"s3://{default_bucket}/input/code/evaluate.py",
        property_files=[evaluation_report],
    )
    
    # Model Creation Step (depends on evaluation completion)
    model = Model(
        image_uri=image_uri,
        model_data=step_tuning.get_top_model_s3_uri(top_k=0, s3_bucket=default_bucket, prefix="output"),
        sagemaker_session=sagemaker_session,
        role=role,
    )
    step_create_model = CreateModelStep(
        name="ChurnCreateModel",
        model=model,
        inputs=CreateModelInput(instance_type="ml.m5.large"),
    )
    
    # Condition Step
    cond_lte = ConditionGreaterThan(
        left=JsonGet(step=step_eval, property_file=evaluation_report, json_path="classification_metrics.auc_score.value"),
        right=0.75,
    )
    step_cond = ConditionStep(
        name="CheckAUCScoreChurnEvaluation",
        conditions=[cond_lte],
        if_steps=[step_create_model],
        else_steps=[],
    )
    
    # Pipeline Instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            processing_instance_type,
            processing_instance_count,
            training_instance_type,
            input_data,
            batch_data,
            use_spot_instances,
            max_run,
            max_wait
        ],
        steps=[step_process, step_tuning, step_eval, step_cond],
        sagemaker_session=sagemaker_session,
    )
    return pipeline
