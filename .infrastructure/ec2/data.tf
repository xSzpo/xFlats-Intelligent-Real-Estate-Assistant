data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

data "terraform_remote_state" "iam" {
  backend = "s3"
  config = {
    bucket = "terraform-274181059559"
    key    = "state/iam.tfstate"
    region = "eu-central-1"
  }
}

data "terraform_remote_state" "secrets" {
  backend = "s3"
  config = {
    bucket = "terraform-274181059559"
    key    = "state/secrets.tfstate"
    region = "eu-central-1"
  }
}