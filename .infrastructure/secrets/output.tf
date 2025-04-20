output "chromedb_token" {
  sensitive = true
  value = {
    token = random_password.chromedb_token.result
  }
}