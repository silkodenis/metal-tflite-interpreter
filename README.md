# Metal Tensorflow Lite Interpreter

`MetalTFLiteInterpreter` is an Objective-C class designed to run TensorFlow Lite models using Metal for accelerated GPU execution.

## Key Features
- **TensorFlow Lite Support:** Allows using models exported in TensorFlow Lite.
- **Hardware Acceleration:** Uses the Metal delegate to execute the model on a GPU.
- **Easy Integration:** Effortlessly binds input and output buffers.

## Requirements
- Xcode 12.5+ (to support Metal)
- TensorFlow Lite C API
- macOS or iOS with Metal support

## Installation
1. **CocoaPods (Recommended):** Add the TensorFlow Lite C API to your project via CocoaPods. In the `Podfile`, add the following configuration:

    ```ruby
    pod 'TensorFlowLiteC', :subspecs => ['Metal']
    ```

    Then, run:

    ```bash
    pod install
    ```

2. **Cloning:** Clone the repository with the code or copy the necessary files to your project.

3. **Dependency Configuration:** Make sure that TensorFlow Lite and TensorFlowLiteCMetal are correctly added to your project.

## Usage
1. **Initialization:**
Create an instance of `MetalTFLiteInterpreter`, specifying the path to your model:

```objc
NSString *modelPath = @"path/to/your/model.tflite";
MetalTFLiteInterpreter *interpreter = [[MetalTFLiteInterpreter alloc] initWithFilePath:modelPath];
```

2. **Buffer Binding:**
Bind an MTLBuffer to an input or output tensor by specifying the index:

```objc
BOOL success = [interpreter bindMetalBuffer:yourMTLBuffer toTensor:INPUT index:0];
if (!success) {
    NSLog(@"Failed to bind Metal buffer.");
}
```

3. **Invoke Model:**
Call invoke to execute the model:

```objc
BOOL invocationSuccess = [interpreter invoke];
if (!invocationSuccess) {
    NSLog(@"Failed to invoke interpreter.");
}
```

4. **Access Results:**
Access results through the output buffer:

```objc
void *outputBuffer = [interpreter bufferPointerForOutputTensor:0];
size_t outputSize = [interpreter bufferSizeForOutputTensor:0];
```

## License
This project is licensed under the [Apache License, Version 2.0](LICENSE).
