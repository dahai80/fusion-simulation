class FusionSimulation < Formula
  desc "Robot virtual simulation training and testing platform for Apple Silicon"
  homepage "https://github.com/dahai80/fusion-simulation"
  url "https://github.com/dahai80/fusion-simulation/releases/download/v0.1.13/fs-release.tar.gz"
  sha256 "81f8fffc7ba346ab0e11d24f1f099b7f926b113677083b4aa111ea7532bbe6fd"
  license "Apache-2.0"
  head "https://github.com/dahai80/fusion-simulation.git", branch: "main"

  depends_on "python@3.12"

  on_macos do
    depends_on macos: :ventura
  end

  def install
    venv = libexec/"venv"
    system "python3.12", "-m", "venv", venv
    venv_pip = venv/"bin/pip"
    system venv_pip, "install", "--upgrade", "pip", "setuptools", "wheel"
    # Pin the resolver-prone web stack: pip's backtracker wanders into fastapi
    # versions that pin anyio<4 (0.106/0.107) and dies with ResolutionImpossible.
    # Versions mirror the verified workspace venv set (v0.1.13, 2026-11-16).
    system venv_pip, "install",
      "fastapi==0.139.2", "starlette==1.3.1", "anyio==4.14.2",
      "httpx==0.28.1", "httpcore==1.0.9", "h11==0.16.0",
      "pydantic==2.13.4", "uvicorn==0.51.0",
      "fusion-core==0.12.1", "setproctitle==1.3.8",
      "numpy>=1.26.0", "pillow>=10.0.0", "defusedxml>=0.7.0",
      "grpcio>=1.60.0", "grpcio-tools>=1.60.0"
    system venv_pip, "install", "--no-deps", "."
    (bin/"fusion-sim").write_env_script venv/"bin/fusion-sim",
      PATH: "#{venv/"bin"}:$PATH"
  end

  test do
    assert_match "v#{version}", shell_output("#{bin}/fusion-sim version")
  end
end
