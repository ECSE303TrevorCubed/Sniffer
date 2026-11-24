{
  stdenv,
  typst,
  ...
}:
stdenv.mkDerivation {
  pname = "sniffer_report";
  version = "0.0.1";
  src = ../report;
  nativeBuildInputs = [ typst ];
  buildPhase = ''
    runHook preBuild
    typst compile report.typ report.pdf
    runHook postBuild
  '';
  installPhase = ''
    runHook preInstall
    install -Dm644 report.pdf "$out/sniffer_report.pdf"
    runHook postInstall
  '';
}
