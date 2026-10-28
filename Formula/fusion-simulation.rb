class FusionSimulation < Formula
  desc "Robot virtual simulation training and testing platform for Apple Silicon"
  homepage "https://github.com/dahai80/fusion-simulation"
  url "https://github.com/dahai80/fusion-simulation/releases/download/v0.1.10/fs-release.tar.gz"
  sha256 "83e717e6c96918378fc183f827979d44cfc7e8a56da573c8df026fc04468db7f"
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
    system venv_pip, "install", "."
    (bin/"fusion-sim").write_env_script venv/"bin/fusion-sim",
      PATH: "#{venv/"bin"}:$PATH"
  end

  test do
    assert_match "v#{version}", shell_output("#{bin}/fusion-sim version")
  end
end
