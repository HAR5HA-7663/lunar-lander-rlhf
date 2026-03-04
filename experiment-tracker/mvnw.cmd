@REM ----------------------------------------------------------------------------
@REM Licensed to the Apache Software Foundation (ASF) under one or more
@REM contributor license agreements. See the NOTICE file distributed with
@REM this work for additional information regarding copyright ownership.
@REM The ASF licenses this file to You under the Apache License, Version 2.0
@REM ----------------------------------------------------------------------------

@IF "%__MVNW_ARG0_NAME__%"=="" (SET "__MVNW_ARG0_NAME__=%~nx0")
@SET MAVEN_PROJECTBASEDIR=%~dp0

@IF NOT "%MAVEN_HOME%"=="" GOTO skipHomeSetting
@SET MAVEN_HOME=%USERPROFILE%\.m2
:skipHomeSetting

@SET MAVEN_WRAPPER_JAR=%MAVEN_HOME%\wrapper\dists\apache-maven-3.9.6-bin\maven-wrapper.jar

@IF EXIST "%MAVEN_WRAPPER_JAR%" GOTO runWrapper

@echo Downloading Maven Wrapper...
@IF NOT EXIST "%MAVEN_HOME%\wrapper\dists\apache-maven-3.9.6-bin" MKDIR "%MAVEN_HOME%\wrapper\dists\apache-maven-3.9.6-bin"
@curl -fsSL -o "%MAVEN_WRAPPER_JAR%" "https://repo.maven.apache.org/maven2/org/apache/maven/wrapper/maven-wrapper/3.2.0/maven-wrapper-3.2.0.jar"

:runWrapper
@java -classpath "%MAVEN_WRAPPER_JAR%" ^
  "-Dmaven.multiModuleProjectDirectory=%MAVEN_PROJECTBASEDIR%" ^
  org.apache.maven.wrapper.MavenWrapperMain ^
  -f "%MAVEN_PROJECTBASEDIR%pom.xml" ^
  %*
