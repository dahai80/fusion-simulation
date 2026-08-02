class FusionSimulation < Formula
  desc "Robot virtual simulation training and testing platform for Apple Silicon"
  homepage "https://github.com/user/fusion-simulation"
  url "https://github.com/user/fusion-simulation/archive/refs/tags/v0.1.1.tar.gz"
  sha256 "PLACEHOLDER_REPLACE_WITH_REAL_SHA256"
  license "MIT"
  head "https://github.com/user/fusion-simulation.git", branch: "main"

  depends_on "python@3.12"

  on_macos do
    depends_on :macos => :ventura
  end

  def install
    venv = libexec/"venv"
    system "python3.12", "-m", "venv", venv
    venv_pip = venv/"bin/pip"
    system venv_pip, "install", "--upgrade", "pip"
    system venv_pip, "install", "."
    (bin/"fusion-sim").write_env_script venv/"bin/fusion-sim",
      PATH: "#{venv/"bin"}:$PATH"
  end

  test do
    system bin/"fusion-sim", "version"
  end
end
