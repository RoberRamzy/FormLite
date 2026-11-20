// React Native 0.83.1's bundled resolver references a constant removed by Gradle 9.
const fs = require('node:fs');
const path = require('node:path');
const settings = path.join(__dirname,
  '../node_modules/@react-native/gradle-plugin/settings.gradle.kts');
const text = fs.readFileSync(settings, 'utf8');
const old = 'id("org.gradle.toolchains.foojay-resolver-convention").version("0.5.0")';
const compatible = 'id("org.gradle.toolchains.foojay-resolver-convention").version("1.0.0")';
if (text.includes(old)) fs.writeFileSync(settings, text.replace(old, compatible));
