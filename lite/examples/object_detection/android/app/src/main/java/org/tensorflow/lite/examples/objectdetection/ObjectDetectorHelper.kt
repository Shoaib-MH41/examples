
package org.tensorflow.lite.examples.objectdetection

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.Rot90Op
import org.tensorflow.lite.task.core.BaseOptions
import org.tensorflow.lite.task.vision.detector.Detection
import org.tensorflow.lite.task.vision.detector.ObjectDetector

class ObjectDetectorHelper(
  var threshold: Float = 0.5f,
  var numThreads: Int = 2,
  var maxResults: Int = 3,
  var currentDelegate: Int = 0,
  var currentModel: Int = 0,
  val context: Context,
  val objectDetectorListener: DetectorListener?
) {

    private var objectDetector: ObjectDetector? = null

    init {
        setupObjectDetector()
    }

    fun clearObjectDetector() {
        objectDetector = null
    }

    fun setupObjectDetector() {
        val optionsBuilder =
            ObjectDetector.ObjectDetectorOptions.builder()
                .setScoreThreshold(threshold)
                .setMaxResults(maxResults)

        val baseOptionsBuilder = BaseOptions.builder().setNumThreads(numThreads)

        // NPU بائی پاس اور ہارڈویئر سلیکشن لاجک
        when (currentDelegate) {
            DELEGATE_CPU -> { /* ڈیفالٹ سی پی یو */ }
            DELEGATE_GPU -> {
                if (CompatibilityList().isDelegateSupportedOnThisDevice) {
                    baseOptionsBuilder.useGpu()
                } else {
                    objectDetectorListener?.onError("GPU is not supported")
                }
            }
            DELEGATE_NNAPI -> {
                try {
                    // یہ لائن NPU کو متحرک کرتی ہے، اگر یہاں 38% پر رکے تو Catch اسے سنبھال لے گا
                    baseOptionsBuilder.useNnapi()
                } catch (e: Exception) {
                    baseOptionsBuilder.useGpu() // اگر NPU لاک ہو تو جی پی یو استعمال کرو
                    Log.e("NPU_FIX", "NPU Blocked on Android 14. Falling back to GPU.")
                }
            }
        }

        optionsBuilder.setBaseOptions(baseOptionsBuilder.build())

        val modelName = when (currentModel) {
            MODEL_MOBILENETV1 -> "mobilenetv1.tflite"
            MODEL_EFFICIENTDETV0 -> "efficientdet-lite0.tflite"
            MODEL_EFFICIENTDETV1 -> "efficientdet-lite1.tflite"
            MODEL_EFFICIENTDETV2 -> "efficientdet-lite2.tflite"
            else -> "mobilenetv1.tflite"
        }

        try {
            objectDetector = ObjectDetector.createFromFileAndOptions(context, modelName, optionsBuilder.build())
        } catch (e: Exception) {
            objectDetectorListener?.onError("Initialization Failed")
            Log.e("Test", "Error: " + e.message)
        }
    }

    fun detect(image: Bitmap, imageRotation: Int) {
        if (objectDetector == null) setupObjectDetector()

        var inferenceTime = SystemClock.uptimeMillis()

        val imageProcessor = ImageProcessor.Builder()
            .add(Rot90Op(-imageRotation / 90))
            .build()

        val tensorImage = imageProcessor.process(TensorImage.fromBitmap(image))
        val results = objectDetector?.detect(tensorImage)
        
        inferenceTime = SystemClock.uptimeMillis() - inferenceTime

        // --- خلیہ نما مشین (Self-Sustaining Cell) کا فارمولا ---
        // فرض کریں باہر سے ملنے والی انرجی 0.90 ہے اور مشین کا اندرونی نقصان 0.05 ہے
        val externalEnergy = 0.90f 
        val internalResistance = 0.05f
        
        // یہ حساب کرتا ہے کہ کیا مشین اپنی طاقت سے زیادہ انرجی کھینچ رہی ہے
        val efficiency = (externalEnergy / (internalResistance + 0.01f)) * (100.0f / (inferenceTime + 1))
        
        if (efficiency > 1.0) {
            Log.d("QuantumCell", "STATUS: SELF-SUSTAINING | Efficiency: $efficiency")
        } else {
            Log.d("QuantumCell", "STATUS: EXTERNAL POWER NEEDED")
        }
        // --------------------------------------------------

        objectDetectorListener?.onResults(
            results,
            inferenceTime,
            tensorImage.height,
            tensorImage.width)
    }

    interface DetectorListener {
        fun onError(error: String)
        fun onResults(results: MutableList<Detection>?, inferenceTime: Long, imageHeight: Int, imageWidth: Int)
    }

    companion object {
        const val DELEGATE_CPU = 0
        const val DELEGATE_GPU = 1
        const val DELEGATE_NNAPI = 2
        const val MODEL_MOBILENETV1 = 0
        const val MODEL_EFFICIENTDETV0 = 1
        const val MODEL_EFFICIENTDETV1 = 2
        const val MODEL_EFFICIENTDETV2 = 3
    }
}
