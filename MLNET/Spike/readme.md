# Installation instructions

First set up a .NET console project:

```shell
$ dotnet new console -o Spike
```

Then add the required Nuget packages:

```shell
$ dotnet add package Microsoft.ML
$ dotnet add package Microsoft.ML.TimeSeries
$ dotnet add package PLplot
```

Make sure PLplot is installed on OS/X:

```shell
$ brew install plplot
```

For installation instructions on other platforms, see: https://surban.github.io/PLplotNet/

We're using PLplotNet as our plotting library. See: https://github.com/surban/PLplotNet

