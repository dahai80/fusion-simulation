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
    system venv_pip, "install", "--upgrade", "pip", "setuptools", "wheel"
    # fusion-core is not on PyPI: pre-install from git so `pip install .` can resolve fusion-core>=0.9.0
    # Clone with retries: pip's internal git clone cannot recover from transient 443 failures
    core_src = libexec/"fusion-core-src"
    cloned = false
    # Try HTTPS first (works everywhere); fall back to SSH (robust when github.com:443 is flaky)
    ["https://github.com/dahai80/fusion-core.git",
     "git@github.com:dahai80/fusion-core.git"].each do |repo_url|
      2.times do
        rm_rf(core_src)
        cloned = system "git", "clone", "--depth", "1", "--branch", "v0.10.0", repo_url, core_src
        break if cloned
        sleep 10
      end
      break if cloned
    end
    odie "git clone of fusion-core failed after 4 attempts" unless cloned
    # --no-build-isolation avoids pip's isolated-build index fetch for setuptools (flaky under brew env)
    system venv_pip, "install", "--no-build-isolation", core_src
    system venv_pip, "install", "--no-build-isolation", "."
    (bin/"fusion-sim").write_env_script venv/"bin/fusion-sim",
      PATH: "#{venv/"bin"}:$PATH"
  end

  test do
    assert_match "v#{version}", shell_output("#{bin}/fusion-sim version")
  end
end
