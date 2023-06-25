import nextcloud_client
from version_util_python.version_util import VersionUtil


def upload_ml_models():
    """
    Upload the ML models to a remote repository
    """
    # Initialize version util class
    versions_lib = VersionUtil()
    print(f"Uploading ML models version {versions_lib.model_training_version}...")

    # Initialize remote repository client
    repository_url = "https://liv.nl.tab.digital/s/TyPqR5HCjExqNQq"
    nc = nextcloud_client.Client.from_public_link(repository_url)

    # Upload models
    try:
        nc.mkdir(versions_lib.model_training_version)
    except nextcloud_client.nextcloud_client.HTTPResponseError:
        print(f"Remote folder {versions_lib.model_training_version} already exists, let's update the ML models.")

    nc.put_file(
        f"/{versions_lib.model_training_version}/c1_BoW_Sentiment_Model.pkl",
        "./ml_models/c1_BoW_Sentiment_Model.pkl"
    )
    nc.put_file(
        f"/{versions_lib.model_training_version}/c2_Classifier_Sentiment_Model",
        "./ml_models/c2_Classifier_Sentiment_Model"
    )

    print("Done, the ml_models folder and the ML models were uploaded successfully!")


if __name__ == "__main__":
    upload_ml_models()
