import kfp
from kfp import dsl

def data_process():
    return dsl.ContainerOp(
        name="data-processing",
        image="hhxcsa/mlopsapp:latest",
        command=["python", "src/data_proccessing.py"]
    )

def model_training():
    return dsl.ContainerOp(
        name="model-training",
        image="hhxcsa/mlopsapp:latest",
        command=["python", "src/model_training.py"]
    )

@dsl.pipeline(
    name="mlops-pipeline",
    description="End to end mlops pipeline"
)
def mlops_pipeline():
    data_processing_task = data_process()
    model_training_task = model_training().after(data_processing_task)

if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        mlops_pipeline, "mlops_pipeline.yaml"
    )
