resource "aws_secretsmanager_secret" "chromedb" {
  name = "chrome-db-${data.aws_caller_identity.current.account_id}"
}

resource "aws_secretsmanager_secret_policy" "chromedb" {
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

resource "aws_secretsmanager_secret_version" "chromedb" {
  secret_id = aws_secretsmanager_secret.chromedb.id
  secret_string = jsonencode({
    IP = aws_eip.chroma_ip.public_ip
  })
}
