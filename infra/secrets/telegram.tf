
resource "aws_secretsmanager_secret" "telegram" {
  name = "telegram-${data.aws_caller_identity.current.account_id}"
}

resource "aws_secretsmanager_secret_policy" "telegram_policy" {
  secret_arn = aws_secretsmanager_secret.telegram.arn

  policy = jsonencode({
    Version = "2012-10-17",
    Statement = [
      {
        Sid    = "AllowEC2AccessToTelegramSecret",
        Effect = "Allow",
        Principal = {
          AWS = [
          data.terraform_remote_state.iam.outputs.ec2_chroma_role.arn]
        },
        Action = [
          "secretsmanager:GetSecretValue",
          "secretsmanager:DescribeSecret"
        ],
        Resource = aws_secretsmanager_secret.telegram.arn
      }
    ]
  })
}
