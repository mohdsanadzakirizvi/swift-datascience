%include "MNIST.swift"

import TensorFlow

let epochCount = 100
let batchSize = 128

// Load dataset
let dataset = MNIST(batchSize: batchSize)

// The LeNet-5 model
var classifier = Sequential {
    Conv2D<Float>(filterShape: (5, 5, 1, 6), padding: .same, activation: relu)
    AvgPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
    Conv2D<Float>(filterShape: (5, 5, 6, 16), activation: relu)
    AvgPool2D<Float>(poolSize: (2, 2), strides: (2, 2))
    Flatten<Float>()
    Dense<Float>(inputSize: 400, outputSize: 120, activation: relu)
    Dense<Float>(inputSize: 120, outputSize: 84, activation: relu)
    Dense<Float>(inputSize: 84, outputSize: 10, activation: softmax)
}

// Using Gradient Descent as the optimizer
let optimizer = SGD(for: classifier, learningRate: 0.1)

print("Beginning training...")

struct Statistics {
    var correctGuessCount: Int = 0
    var totalGuessCount: Int = 0
    var totalLoss: Float = 0
}

// The training loop.
for epoch in 1...epochCount {
    var trainStats = Statistics()
    var testStats = Statistics()

    // Set context to training 
    Context.local.learningPhase = .training

    for i in 0 ..< dataset.trainingSize / batchSize {
        // Get mini-batches of x and y
        let x = dataset.trainingImages.minibatch(at: i, batchSize: batchSize)
        let y = dataset.trainingLabels.minibatch(at: i, batchSize: batchSize)
        
        // Compute the gradient with respect to the model.
        let 𝛁model = classifier.gradient { classifier -> Tensor<Float> in
            let ŷ = classifier(x)
            let correctPredictions = ŷ.argmax(squeezingAxis: 1) .== y
            
            trainStats.correctGuessCount += Int(Tensor<Int32>(correctPredictions).sum().scalarized())
            trainStats.totalGuessCount += batchSize
            
            let loss = softmaxCrossEntropy(logits: ŷ, labels: y)
            trainStats.totalLoss += loss.scalarized()
            
            return loss
        }
        
        // Update the model's differentiable variables along the gradient vector.
        optimizer.update(&classifier, along: 𝛁model)
    }

    // Set context to inference
    Context.local.learningPhase = .inference

    for i in 0 ..< dataset.testSize / batchSize {
        let x = dataset.testImages.minibatch(at: i, batchSize: batchSize)
        let y = dataset.testLabels.minibatch(at: i, batchSize: batchSize)

        // Compute loss on test set
        let ŷ = classifier(x)
        let correctPredictions = ŷ.argmax(squeezingAxis: 1) .== y

        testStats.correctGuessCount += Int(Tensor<Int32>(correctPredictions).sum().scalarized())
        testStats.totalGuessCount += batchSize

        let loss = softmaxCrossEntropy(logits: ŷ, labels: y)

        testStats.totalLoss += loss.scalarized()
    }

    let trainAccuracy = Float(trainStats.correctGuessCount) / Float(trainStats.totalGuessCount)
    let testAccuracy = Float(testStats.correctGuessCount) / Float(testStats.totalGuessCount)

    print("""
          [Epoch \(epoch)] \
          Training Loss: \(trainStats.totalLoss), \
          Training Accuracy: \(trainStats.correctGuessCount)/\(trainStats.totalGuessCount) \
          (\(trainAccuracy)), \
          Test Loss: \(testStats.totalLoss), \
          Test Accuracy: \(testStats.correctGuessCount)/\(testStats.totalGuessCount) \
          (\(testAccuracy))
          """)
}
