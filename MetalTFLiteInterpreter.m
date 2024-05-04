/*
 * Copyright (c) [2024] [Denis Silko]
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#import "MetalTFLiteInterpreter.h"
#import <TensorFlowLiteC/TensorFlowLiteC.h>
#import <TensorFlowLiteCMetal/TensorFlowLiteCMetal.h>

@interface MetalTFLiteInterpreter()

@property (assign, nonatomic) TfLiteModel *model;
@property (assign, nonatomic) TfLiteDelegate *delegate;
@property (assign, nonatomic) TfLiteInterpreter *interpreter;

@end

@implementation MetalTFLiteInterpreter

- (instancetype)initWithFilePath:(NSString *)filePath {
    if (self = [super init]) {
        self.delegate = TFLGpuDelegateCreate(NULL);
        self.model = TfLiteModelCreateFromFile([filePath UTF8String]);
        
        TfLiteInterpreterOptions *options = TfLiteInterpreterOptionsCreate();
        self.interpreter = TfLiteInterpreterCreate(_model, options);
        
        if (TfLiteInterpreterModifyGraphWithDelegate(_interpreter, _delegate) != kTfLiteOk) {
            NSLog(@"MetalTFLiteInterpreter Error - ModifyGraphWithDelegate: Unable to modify graph with delegate.");
            return nil;
        }
        
        TfLiteSetAllowBufferHandleOutput(_interpreter, true);
        
        if (TfLiteInterpreterAllocateTensors(_interpreter) != kTfLiteOk) {
            NSLog(@"MetalTFLiteInterpreter Error - AllocateTensors: Failed to allocate tensors.");
            return nil;
        }
    }
    
    return self;
}
    
- (void)dealloc {
    TfLiteModelDelete(_model);
    TFLGpuDelegateDelete(_delegate);
    TfLiteInterpreterDelete(_interpreter);
}

// MARK: - Public

- (BOOL)bindMetalBuffer:(id<MTLBuffer>)buffer toTensor:(IOTensorType)type index:(int)index {
    int tensorIndex = 0;
    
    switch (type) {
        case INPUT:
            tensorIndex = TfLiteInterpreterGetInputTensorIndex(_interpreter, index);
            break;
            
        case OUTPUT:
            tensorIndex = TfLiteInterpreterGetOutputTensorIndex(_interpreter, index);
            break;
    }
    
    if (!TFLGpuDelegateBindMetalBufferToTensor(_delegate, tensorIndex, buffer)) {
        NSLog(@"MetalTFLiteInterpreter Error - Bind: Could not bind buffer %@ to %@ tensor (%d)", 
              buffer, [self.class tensorTypeName:type], index);
        
        return NO;
    }
    
    NSLog(@"MetalTFLiteInterpreter Success - Buffer %@ successfully bound to %@ tensor (%d).", 
          buffer, [self.class tensorTypeName:type], index);
    
    return YES;
}

- (BOOL)invoke {
    if (TfLiteInterpreterInvoke(_interpreter) != kTfLiteOk) {
        NSLog(@"MetalTFLiteInterpreter Error - Invoke: Failed to invoke interpreter due to unknown error.");
        return NO;
    }
    
    return YES;
}

- (void *)bufferPointerForOutputTensor:(int)index {
    const TfLiteTensor *tensor = TfLiteInterpreterGetOutputTensor(_interpreter, index);
    
    return TfLiteTensorData(tensor);
}

- (size_t)bufferSizeForOutputTensor:(int)index {
    const TfLiteTensor *tensor = TfLiteInterpreterGetOutputTensor(_interpreter, index);
    
    return TfLiteTensorByteSize(tensor);
}

// MARK: - Private

+ (NSString *)tensorTypeName:(IOTensorType)type {
    return type == INPUT ? @"input" : @"output";
}

@end
