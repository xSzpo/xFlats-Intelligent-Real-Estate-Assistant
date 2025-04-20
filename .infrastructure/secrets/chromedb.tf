resource "random_password" "chromedb_token" {
  length           = 32
  override_special = "!@#$%*=+"
  special          = true
}

resource "aws_secretsmanager_secret" "chromedb" {
  name = "chromedb-${data.aws_caller_identity.current.account_id}"
}

resource "aws_secretsmanager_secret_policy" "chromedb_policy" {
  secret_arn = aws_secretsmanager_secret.chromedb.arn

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Sid    = "AllowEC2AccessTochromedbSecret",
        Effect = "Allow",
        Principal = {
          AWS = [
            data.terraform_remote_state.iam.outputs.ec2_chroma_role.arn
          ]
        },
        Action = [
          "secretsmanager:GetSecretValue",
          "secretsmanager:DescribeSecret"
        ],
        Resource = aws_secretsmanager_secret.chromedb.arn
      }
    ]
  })
}

resource "aws_secretsmanager_secret_version" "chromedb_token" {
  secret_id = aws_secretsmanager_secret.chromedb.id
  secret_string = jsonencode({
    TOKEN = random_password.chromedb_token.result
  })
}
