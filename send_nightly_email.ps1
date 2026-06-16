# send_nightly_email.ps1
# Reads NIGHTLY_SUMMARY.txt and emails it to Shlomi via Gmail.
# Schedule this as a Windows Task at 7:00am daily.
# The nightly agent writes NIGHTLY_SUMMARY.txt; this script delivers it.

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$summaryFile = Join-Path $scriptDir "NIGHTLY_SUMMARY.txt"
$credFile = Join-Path $scriptDir ".email_credentials"

# Read credentials
if (-not (Test-Path $credFile)) {
    Write-Error "Missing .email_credentials at $credFile"
    exit 1
}
$creds = Get-Content $credFile
$fromEmail = $creds[0].Trim()
$appPassword = $creds[1].Trim()

# Read summary — latest entry only, UTF-8 so emoji/dashes render correctly
if (Test-Path $summaryFile) {
    $allLines = Get-Content $summaryFile -Encoding UTF8
    # Find the last "=== YYYY-MM-DD" separator line and take everything from there
    $lastSepIndex = -1
    for ($i = $allLines.Length - 1; $i -ge 0; $i--) {
        if ($allLines[$i] -match '^=== \d{4}-\d{2}-\d{2}') {
            $lastSepIndex = $i
            break
        }
    }
    if ($lastSepIndex -ge 0) {
        $body = ($allLines[$lastSepIndex..($allLines.Length - 1)] -join "`n").Trim()
    } else {
        $body = ($allLines -join "`n").Trim()
    }
    $subject = "EKG Nightly Agent Summary - $(Get-Date -Format 'yyyy-MM-dd')"
} else {
    $body = "No NIGHTLY_SUMMARY.txt found. The nightly agent may not have run."
    $subject = "EKG Nightly Agent - No summary ($(Get-Date -Format 'yyyy-MM-dd'))"
}

# Send via Gmail SMTP (port 587, STARTTLS)
$smtpServer = "smtp.gmail.com"
$smtpPort = 587
$toEmail = $fromEmail

try {
    $smtp = New-Object System.Net.Mail.SmtpClient($smtpServer, $smtpPort)
    $smtp.EnableSsl = $true
    $smtp.UseDefaultCredentials = $false  # must be set BEFORE Credentials
    $smtp.Credentials = New-Object System.Net.NetworkCredential($fromEmail, $appPassword)

    $mail = New-Object System.Net.Mail.MailMessage
    $mail.From = $fromEmail
    $mail.To.Add($toEmail)
    $mail.Subject = $subject
    $mail.Body = $body

    $smtp.Send($mail)
    Write-Host "Email sent successfully: $subject"
} catch {
    Write-Error "Failed to send email: $_"
    exit 1
}
