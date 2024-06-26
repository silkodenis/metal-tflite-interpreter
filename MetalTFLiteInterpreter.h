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

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

NS_ASSUME_NONNULL_BEGIN

typedef NS_ENUM(NSUInteger, IOTensorType) {
    INPUT,
    OUTPUT
};

@interface MetalTFLiteInterpreter : NSObject

- (instancetype)initWithFilePath:(NSString *)filePath;

- (void *)bufferPointerForOutputTensor:(int)index;
- (size_t)bufferSizeForOutputTensor:(int)index;
- (BOOL)bindMetalBuffer:(id<MTLBuffer>)buffer toTensor:(IOTensorType)type index:(int)index;
- (BOOL)invoke;

@end

NS_ASSUME_NONNULL_END
