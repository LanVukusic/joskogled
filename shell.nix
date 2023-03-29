{ pkgs ? import <nixpkgs> {}}:
let
  my-python-packages = p: with p; [
      jupyter
      ipykernel
      ipython
      numpy
      torch
      opencv4
      matplotlib
      # other python packages
    ];
    my-python = pkgs.python3.withPackages my-python-packages;
in
pkgs.mkShell {
  name="RIS 2023 tekmovanje";
  
  
  buildInputs = [
    pkgs.python3
    my-python
  ];
  shellHook = ''
    echo -e " ❄️ RIS 2023"
    echo -e " ❄️ Za zagon na slingu uporabli.  \033[0;36m $ sbatch run.sh \033[0m"
  '';
} 