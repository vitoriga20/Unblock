plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

// 用独立构建目录，避免被占用的输出文件阻塞增量构建
layout.buildDirectory.set(file("build-app"))

android {
    namespace = "com.unblock.app"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.unblock.app"
        minSdk = 28
        targetSdk = 34
        versionCode = 1
        versionName = "0.1.0-m1"
        ndk {
            abiFilters += listOf("arm64-v8a", "armeabi-v7a")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    androidResources {
        noCompress += listOf("onnx")
    }

    packaging {
        resources.excludes += setOf("META-INF/DEPENDENCIES", "META-INF/LICENSE*", "META-INF/NOTICE*")
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.6")
    implementation("org.opencv:opencv:4.10.0")
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.19.2")
}
