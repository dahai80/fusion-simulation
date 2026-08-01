class FusionSimulation < Formula
  desc "Robot virtual simulation training and testing platform for Apple Silicon"
  homepage "https://github.com/user/fusion-simulation"
  url "https://github.com/user/fusion-simulation/archive/refs/tags/v0.1.0.tar.gz"
  sha256 "PLACEHOLDER_SHA256"
  license "MIT"

  depends_on "python@3.12"

  resource "pybullet" do
    url "https://files.pythonhosted.org/packages/source/p/pybullet/pybullet-3.2.7.tar.gz"
    sha256 "PLACEHOLDER_SHA256"
  end

  resource "httpx" do
    url "https://files.pythonhosted.org/packages/source/h/httpx/httpx-0.28.1.tar.gz"
    sha256 "PLACEHOLDER_SHA256"
  end

  resource "grpcio" do
    url "https://files.pythonhosted.org/packages/source/g/grpcio/grpcio-1.68.1.tar.gz"
    sha256 "PLACEHOLDER_SHA256"
  end

  resource "protobuf" do
    url "https://files.pythonhosted.org/packages/source/p/protobuf/protobuf-5.29.3.tar.gz"
    sha256 "PLACEHOLDER_SHA256"
  end

  def install
    virtualenv_install_with_resources
  end

  test do
    system bin/"fusion-simulation", "version"
  end
end
