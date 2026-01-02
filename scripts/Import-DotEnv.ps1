[CmdletBinding()]
param(
    # 第1引数で .env パスを受け取れる（例: .\Import-DotEnv.ps1 .\.env）
    [Parameter(Position = 0, Mandatory = $false)]
    [Alias("Path")]
    [string]$EnvFile = ".env",

    # 既に存在する環境変数を上書きするか
    [Parameter(Mandatory = $false)]
    [switch]$Override
)

function Remove-UnquotedInlineComment([string]$s) {
    # 「クオート外の #」をコメント開始として切り捨て（安全寄り）
    # ただし、# を値として使いたい場合は "..." または '...' で囲むこと。
    $inSingle = $false
    $inDouble = $false

    for ($i = 0; $i -lt $s.Length; $i++) {
        $ch = $s[$i]

        if ($ch -eq "'" -and -not $inDouble) { $inSingle = -not $inSingle; continue }
        if ($ch -eq '"' -and -not $inSingle) { $inDouble = -not $inDouble; continue }

        if (-not $inSingle -and -not $inDouble -and $ch -eq "#") {
            # 直前が空白 or 行頭ならコメント扱い（よくある .env の慣習）
            if ($i -eq 0 -or [char]::IsWhiteSpace($s[$i - 1])) {
                return $s.Substring(0, $i).TrimEnd()
            }
        }
    }

    return $s
}

function Unquote-EnvValue([string]$value) {
    $v = $value.Trim()

    if ($v.Length -ge 2 -and $v.StartsWith('"') -and $v.EndsWith('"')) {
        $inner = $v.Substring(1, $v.Length - 2)

        # ダブルクオート内の代表的なエスケープだけを“文字として”解釈（コード評価はしない）
        $inner = $inner.Replace('\n', "`n").Replace('\r', "`r").Replace('\t', "`t")
        $inner = $inner.Replace('\"', '"').Replace('\\', '\')
        return $inner
    }

    if ($v.Length -ge 2 -and $v.StartsWith("'") -and $v.EndsWith("'")) {
        # シングルクオートは基本リテラル扱い
        return $v.Substring(1, $v.Length - 2)
    }

    return $v
}

if (-not (Test-Path -LiteralPath $EnvFile)) {
    throw "Dotenv file not found: $EnvFile"
}

Get-Content -LiteralPath $EnvFile | ForEach-Object {
    $raw = $_

    # トリムして空行・コメント行をスキップ
    $line = $raw.Trim()
    if (-not $line) { return }
    if ($line.StartsWith("#")) { return }

    # export を許容
    if ($line.StartsWith("export ")) {
        $line = $line.Substring(7).Trim()
    }

    # 行末コメントを削る（クオート外のみ）
    $line = Remove-UnquotedInlineComment $line
    if (-not $line) { return }

    # KEY=VALUE をパース
    if ($line -notmatch '^(?<key>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?<val>.*)$') {
        # 必要なら警告を出したい場合は Write-Warning に変更可
        return
    }

    $name = $Matches['key']
    $value = Unquote-EnvValue $Matches['val']

    # 既存があるなら上書きしない（Override があれば上書き）
    if (-not $Override -and (Test-Path "Env:\$name")) { return }

    Set-Item -Path "Env:\$name" -Value $value
}
