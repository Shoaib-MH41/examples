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

    // --- سائنسی ڈیٹا ریکارڈنگ کے لیے نئے متغیرات ---
    private var minInferenceTime = Long.MAX_VALUE
    private var maxInferenceTime = 0L
    private var stableEfficiency = 0.0f

    init {
        setupObjectDetector()
    }

    fun clearObjectDetector() {
        objectDetector = null
        // ری سیٹ کرنے پر پرانا ڈیٹا صاف کریں
        minInferenceTime = Long.MAX_VALUE
        maxInferenceTime = 0L
    }

    fun setupObjectDetector() {
        val optionsBuilder =
            ObjectDetector.ObjectDetectorOptions.builder()
                .setScoreThreshold(threshold)
                .setMaxResults(maxResults)

        val baseOptionsBuilder = BaseOptions.builder().setNumThreads(numThreads)

        when (currentDelegate) {
            DELEGATE_CPU -> { /* Default */ }
            DELEGATE_GPU -> {
                if (CompatibilityList().isDelegateSupportedOnThisDevice) {
                    baseOptionsBuilder.useGpu()
                } else {
                    objectDetectorListener?.onError("GPU is not supported")
                }
            }
            DELEGATE_NNAPI -> {
                try {
                    baseOptionsBuilder.useNnapi()
                } catch (e: Exception) {
                    baseOptionsBuilder.useGpu()
                    Log.e("NPU_FIX", "NPU Blocked. Falling back to GPU.")
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

        // --- صلاحیت کا تجزیہ (Capability Analysis) ---
        
        // سب سے بہترین رفتار (Min Time) ریکارڈ کریں
        if (inferenceTime < minInferenceTime && inferenceTime > 0) {
            minInferenceTime = inferenceTime
        }
        
        // سب سے زیادہ بوجھ (Max Time) ریکارڈ کریں
        if (inferenceTime > maxInferenceTime) {
            maxInferenceTime = inferenceTime
        }

        // کوانٹم انرجی کا مستحکم حساب
        val externalEnergy = 0.95f 
        val internalResistance = 0.05f
        
        // اگر وقت کم ہے تو صلاحیت زیادہ ہے
        val currentEfficiency = (externalEnergy / (internalResistance + 0.01f)) * (100.0f / (inferenceTime + 1))
        
        // لاگز میں مستقل ریکارڈ (یہ نمبرز بھاگیں گے نہیں بلکہ صلاحیت دکھائیں گے)
        Log.d("QuantumLab", """
            [REPORT] 
            Current: $inferenceTime ms | Best (Min): $minInferenceTime ms | Worst (Max): $maxInferenceTime ms
            Efficiency Score: $currentEfficiency
        """.trimIndent())

        // رزلٹ بھیجتے وقت ہم سب سے بہترین اسپیڈ بھی پاس کر سکتے ہیں
        objectDetectorListener?.onResults(
            results,
            inferenceTime, // موجودہ وقت
            tensorImage.height,
            tensorImage.width
        )
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
