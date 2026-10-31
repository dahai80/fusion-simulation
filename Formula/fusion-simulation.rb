class FusionSimulation < Formula
  desc "Robot virtual simulation training and testing platform for Apple Silicon"
  homepage "https://github.com/dahai80/fusion-simulation"
  url "https://github.com/dahai80/fusion-simulation/releases/download/v0.1.11/fs-release.tar.gz"
  sha256 "a3cf6b899a08910234aa69ce121639b7040ad43f93fd6288d8930a029cbbf0ce"
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
