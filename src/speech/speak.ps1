# Fixed local speech helper. Message data comes through stdin, never as code.
$ErrorActionPreference = 'Stop'
$taskSpeaker = $null
try {
    Add-Type -AssemblyName System.Speech
    $taskEncoded = [Console]::In.ReadToEnd()
    $taskMessage = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($taskEncoded))
    $taskSpeaker = [System.Speech.Synthesis.SpeechSynthesizer]::new()
    $taskSpeaker.SetOutputToDefaultAudioDevice()
    $taskSpeaker.Speak($taskMessage)
    exit 0
} catch {
    exit 1
} finally {
    if ($null -ne $taskSpeaker) { $taskSpeaker.Dispose() }
}
