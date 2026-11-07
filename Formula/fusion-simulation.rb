class FusionSimulation < Formula
  desc "Robot virtual simulation training and testing platform for Apple Silicon"
  homepage "https://github.com/dahai80/fusion-simulation"
  url "https://github.com/dahai80/fusion-simulation/releases/download/v0.1.12/fs-release.tar.gz"
  sha256 "5fc61aebc2bb764f15f1843dd7b935c7ac642439385a97fea447350abfdf794e"
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
    system venv_pip, "install", "--upgrade", "pip"
    # fusion-core is not on PyPI: pre-install from git so `pip install .` can resolve fusion-core>=0.9.0
    system venv_pip, "install",
      "fusion-core @ git+https://github.com/dahai80/fusion-core.git@v0.10.0"
    system venv_pip, "install", "."
    (bin/"fusion-sim").write_env_script venv/"bin/fusion-sim",
      PATH: "#{venv/"bin"}:$PATH"
  end

  test do
    assert_match "v#{version}", shell_output("#{bin}/fusion-sim version")
  end
end
