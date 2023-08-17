// Create a managed instance group

"@type":            "gcloud.create_mig_from_template"
project:            "zetta-research"
zone:               "us-east1-c"
mig_name:           "ddp-workers"
template_name:      "ddp-workers"
max_replicas:       2
