package ampControl.model.training;

import ampControl.audio.ClassifierInputProviderFactory;
import ampControl.audio.processing.ProcessingFactoryFromString;
import ampControl.audio.processing.ProcessingResult;
import ampControl.model.training.data.*;
import ampControl.model.training.data.iterators.CachingDataSetIterator;
import ampControl.model.training.data.iterators.Cnn2DDataSetIterator;
import ampControl.model.training.data.processing.SilenceProcessor;
import ampControl.model.training.listen.TrainScoreListener;
import ampControl.model.training.model.GenericModelHandle;
import ampControl.model.training.model.GraphModelAdapter;
import ampControl.model.training.model.ModelHandle;
import ampControl.model.visualize.RealTimePlot;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.eval.ROCMultiClass;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * Harness for training and evaluation of {@link ModelHandle ModelHandles}. Models will take turn in doing fitting. Will
 * evaluate models in regular intervals and saves the last evaluated model as long as the accuracy is not worse than 90% 
 * of the best evaluation accuracy for the model. Will also plot accuracy and score for training, eval and best eval.
 * 
 * @author Christian Skärby
 */
public class TrainingHarness {

    private static final Logger log = LoggerFactory.getLogger(TrainingHarness.class);
    
    private static final boolean doStatsLogging = false;
    private static final int maxNrofTrainingSteps = 40000;
    private static final int evalEveryNrofSteps = 40;
    private static final String bestSuffix = "_best";
    private static final double saveThreshold = 0.9;

    private static final String trainEvalPrefix = "Train";
    private static final String lastEvalPrefix = "LastEval";
    private static final String bestEvalPrefix = "BestEval";

    private final List<ModelHandle> modelsToTrain;
    private final String modelSaveDir;
    private RealTimePlot<Integer, Double> evalPlot;
    private RealTimePlot<Integer, Double> scorePlot;

    private static class ModelInfo {
        private int skipEval = 0;
        private int currIter = 0;

        private ModelInfo(ModelHandle mdh) {
            mdh.getModel().addListeners((IterationListener) (model, iteration, epoch) -> currIter = iteration);
        }
    }


    public TrainingHarness(List<ModelHandle> modelsToTrain, String modelSaveDir) {
        this.modelsToTrain = modelsToTrain;
        this.modelSaveDir = modelSaveDir;
        evalPlot = new RealTimePlot<>("Accuracy", modelSaveDir + File.separator + "plots");
        scorePlot = new RealTimePlot<>("Score", modelSaveDir + File.separator + "plots");
        addListeners(modelsToTrain);
        for (ModelHandle md : modelsToTrain) {
            evalPlot.createSeries(trainEvalName(md.name()));
            evalPlot.createSeries(lastEvalName(md.name()));
            evalPlot.createSeries(bestEvalName(md.name()));

            scorePlot.createSeries(trainEvalName(md.name()));
            scorePlot.createSeries(lastEvalName(md.name()));
            scorePlot.createSeries(bestEvalName(md.name()));
            //Evaluation eval = md.model.evaluate(md.evalIter);
            //Evaluation eval = new Evaluation();
            //evaluate(md, eval);
            //md.bestEvalScore = eval.accuracy();
            //md.evalIter.resetCursor();
            // log.info(md.name() + " acc: " + md.getBestEvalScore());
        }
    }

    private void addListeners(List<ModelHandle> models) {

        for (final ModelHandle md : models) {
            if (doStatsLogging) {
                UIServer uiServer = UIServer.getInstance();
                //Alternative: new FileStatsStorage(File) - see UIStorageExample
                StatsStorage statsStorage = new FileStatsStorage(new File(md.name() + "_stats"));
                uiServer.attach(statsStorage);
                int listenerFrequency = md.getNrofBatchesForTraining();
                md.getModel().addListeners(new StatsListener(statsStorage, listenerFrequency));
            }
            final String trainName = trainEvalName(md.name());
            md.getModel().addListeners(new ScoreIterationListener(md.getNrofBatchesForTraining()));
            md.getModel().addListeners(new TrainScoreListener(md.getNrofBatchesForTraining(), (i, s) -> scorePlot.plotData(trainName, i, s)));
            md.createTrainingEvalListener((i, e) -> evalPlot.plotData(trainName, i, e));
        }
    }


    public void startTraining() {
        Nd4j.getMemoryManager().setAutoGcWindow(5000);
        Map<ModelHandle, ModelInfo> modelInfoMap = modelsToTrain.stream()
                .collect(Collectors.toMap(
                        Function.identity(),
                        ModelInfo::new
                ));
        for (int trainingStep = 0; trainingStep < maxNrofTrainingSteps; trainingStep++) {
            log.info("****************************** Training step " + trainingStep + " started! ***************************************");
            for (ModelHandle md : modelsToTrain) {
                printSynchronized("Training model with curr best " + md.getBestEvalScore() + ", name: " + md.name());
                long starttime = System.nanoTime();
                md.fit();
                long endtime = System.nanoTime();
                double time = (endtime - starttime) / 1000000d;
                printSynchronized("Training took " + time + " ms for " + md.getNrofTrainingExamplesPerBatch() + " examples, " + time / (double) md.getNrofTrainingExamplesPerBatch() + " ms per example");
                if (trainingStep % evalEveryNrofSteps == evalEveryNrofSteps - 1) {
                    if (modelInfoMap.get(md).skipEval > 0) {
                        modelInfoMap.get(md).skipEval--;
                        printSynchronized("Skip eval! " + modelInfoMap.get(md).skipEval);
                    } else {
                        printSynchronized("Begin eval of " + md.name());
                        final double prevEvalScore = md.getBestEvalScore();
                        starttime = System.nanoTime();
                        final Evaluation eval = md.createEvalTemplate();
                        final ROCMultiClass roc = new ROCMultiClass();
                        md.eval(eval, roc);

                        //Evaluation eval = new Evaluation();
                        //evaluate(md, eval);
                        endtime = System.nanoTime();
                        final double evalTime = (endtime - starttime) / 1000000d;

                        final double accuracy = eval.accuracy();
                        final double evalScore = md.getModel().score();
                        if (accuracy < 0.7) {
                            modelInfoMap.get(md).skipEval = 3;
                        } else if (accuracy < 0.8) {
                            modelInfoMap.get(md).skipEval = 2;
                        } else if (accuracy < 0.9) {
                            modelInfoMap.get(md).skipEval = 1;
                        }

                        final ModelHandle fmdh = md;
                        final int evalIterNr = modelInfoMap.get(fmdh).currIter;
                        try { // ROC takes pretty long time to compute; put it in background while we eval the next model
                            new Thread(() -> {

                                try {
                                    printSynchronized("Eval report for " + fmdh.name());
                                    printSynchronized(eval.stats());
                                    printSynchronized("\n" + eval.confusionToString());

                                    final String lastEvalLabel = lastEvalName(fmdh.name());
                                    evalPlot.plotData(lastEvalLabel, evalIterNr, accuracy);
                                    evalPlot.storePlotData(lastEvalLabel);
                                    scorePlot.plotData(lastEvalLabel, evalIterNr, evalScore);
                                    scorePlot.storePlotData(lastEvalLabel);
                                    final String trainEvalLabel = trainEvalName(fmdh.name());
                                    evalPlot.storePlotData(trainEvalLabel);
                                    scorePlot.storePlotData(trainEvalLabel);

                                  //  printSynchronized("ROC report for " + fmdh.name() + "\n" + roc.stats());
                                    printSynchronized("Accuracy = " + accuracy + " Best: " + fmdh.getBestEvalScore());
                                    printSynchronized("Evaluation took " + evalTime + " ms for " + fmdh.getNrofEvalExamples() + " examples, " + evalTime / (double) fmdh.getNrofEvalExamples() + " ms per example");
                                } catch (IOException e) {
                                    e.printStackTrace();
                                }
                            }).start();
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                        if (accuracy >= saveThreshold * md.getBestEvalScore()) {
                            try {
                                new Thread(() -> {
                                    try {
                                        synchronized (modelsToTrain) {

                                            String fileBaseName = modelSaveDir + File.separator + md.name().hashCode();
                                            writeModel(md, eval, roc, fileBaseName);
                                            if (accuracy >= prevEvalScore) {
                                                fileBaseName = modelSaveDir + File.separator + md.name().hashCode() + bestSuffix;
                                                String bestLabel = bestEvalName(fmdh.name());
                                                evalPlot.plotData(bestLabel, evalIterNr, accuracy);
                                                evalPlot.storePlotData(bestLabel);
                                                scorePlot.plotData(bestLabel, evalIterNr, evalScore);
                                                scorePlot.storePlotData(bestLabel);
                                                writeModel(md, eval, roc, fileBaseName);
                                                // EvaluationTools.exportRocChartsToHtmlFile(roc, new File(fileBaseName + "_roc.html"));
                                            }
                                        }
                                    } catch (Exception e) {
                                        e.printStackTrace();
                                    }
                                }
                                ).start();

                            } catch (Exception e) {
                                e.printStackTrace();
                            }
                        }
                    }
                }
            }

            for (ModelHandle md : modelsToTrain) {
                md.resetTraining();
            }

        }
    }

//	private void evaluate(ModelData md, Evaluation eval) {
//		while(md.getEvalIter().hasNext()) {
//            DataSet set = md.getEvalIter().next();
//            INDArray output = md.model.output(set.getFeatures());
//            //int timeSeriesLength = timeSeriesOutput.size(0);		//Size of time dimension
//            //INDArray lastTimeStepProbabilities = timeSeriesOutput.get(NDArrayIndex.point(timeSeriesLength-1), NDArrayIndex.all());
//            eval.eval(set.getLabels(), output);
//        }
//	}

    private void writeModel(ModelHandle mdh, Evaluation eval, ROCMultiClass roc, String fileBaseName) throws IOException {
        ModelSerializer.writeModel(mdh.getModel(), new File(fileBaseName), true);
        Path path = Paths.get(fileBaseName + ".score");
        BufferedWriter writer = Files.newBufferedWriter(path);
        writer.write(mdh.name() + "\n");
        writer.write(eval.confusionToString());
        writer.write(eval.stats());
       // writer.write(roc.stats());
        writer.close();
    }

    private synchronized void printSynchronized(String str) {
        log.info(str);
    }

    private String trainEvalName(String modelName) {
        return trainEvalPrefix + modelName.hashCode();
    }

    private String lastEvalName(String modelName) {
        return lastEvalPrefix + modelName.hashCode();
    }

    private String bestEvalName(String modelName) {
        return bestEvalPrefix + modelName.hashCode();
    }

    public static void main(String[] args) {
        DataTypeUtil.setDTypeForContext(DataBuffer.Type.HALF);
        String modelBaseDir = "E:\\Software projects\\java\\leadRythm\\RythmLeadSwitch\\models\\";
        String model = "ws_100_sgpp_spgr_fft_256_olf_8_pipe_lgsc_4x_C_128_3_ELU_BN_w_Mp2_2_t_2x_dnn_512_ReLU_w_d0p0_t_out_w_0p5_0p5_0p8_1p0Nesterovs";
        //"ws_100_fft_512_olf_16_nfs_128_sgpp_lgsc_4x_C_128_3_ELU_BN_w_Mp2_2_t_2x_dnn_512_ReLU_w_d0p5_t_out_w_0p5_0p5_0p8_1p0Nesterovs";
               // "ws_100_fft_512_olf_16_nfs_128_sgpp_lgsc_z3_3_t_3x_rb_C_128_4_ELU_BN_t_se16_ReLU_w_Mp2_2_t_3x_dnn_512_SELU_w_d0p2_t_out_w_0p1_0p3_0p8_1p0";
        //"ws_100_fft_512_olf_16_nfs_128_Cnn2d_4_layers_128_kernels_4_4_mp_2_2_dnn_2_layers_64_dnnW_best_old";
        //String model2 = modelBaseDir + "ws_100_fft_512_olf_16_nfs_128_Cnn2d_4_layers_128_kernels_4_4_mp_2_2_dnn_2_layers_64_dnnW_best";
        //"ws_100_fft_512_olf_16_nfs_128_Cnn2d_4_layers_2_kernelGrowth_16_kernels_4_4_mp_2_2_dnn_2_layers_256_dnnW";
        //"ws_200_fft_1024_olf_16_nfs_128_Cnn2d_4_layers_128_kernels_4_4_mp_2_2_dnn_2_layers_64_dnnW_best";
        int clipLengthMs = 1000;
        int clipSamplingRate = 44100;
        Path baseDir = Paths.get("E:\\Software projects\\python\\lead_rythm\\data");
        List<String> labels = Arrays.asList("silence", "noise", "rythm", "lead");
        int trainingIterations = 20;
        int trainBatchSize = 32;
        int evalBatchSize = 1;
        double evalSetPercentage = 5;

        final ProcessingResult.Factory audioPostProcFactory = new ProcessingFactoryFromString(clipSamplingRate).get(model);
        final int timeWindowSize = ClassifierInputProviderFactory.parseWindowSize(model);
        final SilenceProcessor silence = new SilenceProcessor(clipSamplingRate * clipLengthMs / (1000 / timeWindowSize) / 1000, () -> audioPostProcFactory);
        Map<String, AudioDataProvider.AudioProcessorBuilder> labelToBuilder = new LinkedHashMap<>();
        labelToBuilder.put("silence", () -> silence);
        labelToBuilder = Collections.unmodifiableMap(labelToBuilder);
        MultiplyLabelExpander labelExpander = new MultiplyLabelExpander()
                .addExpansion("noise", 20)
                .addExpansion("rythm", 100)
                .addExpansion("lead", 100);
        final DataProviderBuilder train = new TrainingDataProviderBuilder(labelToBuilder, labelExpander, clipLengthMs, timeWindowSize, () -> audioPostProcFactory, new Random().nextInt());
        final DataProviderBuilder eval = new EvalDataProviderBuilder(labelToBuilder, labelExpander, clipLengthMs, timeWindowSize, () -> audioPostProcFactory, 666);

        try {
            DataSetFileParser.parseFileProperties(baseDir, new TrainingDescription.DataSetMapper(train, eval, evalSetPercentage));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        final CachingDataSetIterator trainIter = new CachingDataSetIterator(
                new Cnn2DDataSetIterator(
                        train.createProvider(), trainBatchSize, labels),
                trainingIterations);

        final int evalCacheSize = (int) (0.75 * (clipLengthMs / timeWindowSize * (eval.getNrofFiles() / evalBatchSize)));
        //final int evalCacheSize = 1;
        final CachingDataSetIterator evalIter = new CachingDataSetIterator(
                new Cnn2DDataSetIterator(eval.createProvider(), evalBatchSize, labels),
                evalCacheSize);


        try {
            final ComputationGraph classifier = ModelSerializer.restoreComputationGraph(modelBaseDir + model.hashCode() +"_best", false);
            GenericModelHandle mdh = new GenericModelHandle(trainIter, evalIter, new GraphModelAdapter(classifier), model, 1);
            log.info("shape: " + Arrays.toString(trainIter.next().getFeatures().shape()));

//new TrainingHarness(Collections.singletonList(mdh), modelBaseDir).startTraining();
            Evaluation evaluation = mdh.createEvalTemplate();
            //ROCMultiClass roc = new ROCMultiClass();
           // mdh.eval(evaluation);
//            for(int i = 0; i< 100; i++) {
//                log.info(i);
//                evalIter.next();
//                evalIter.reset();
//            }
//            for(int i = 0; i< 100; i++) {
//                DataSet ds = evalIter.next();
//                INDArray[] output = classifier.output(ds.getFeatures());
//                log.info("output:  " + output[0]);
//                log.info("label : " + ds.getLabels());
//                evalIter.reset();
//                if(!output[0].argMax().equalsWithEps(ds.getLabels().argMax(),1e-10)) {
//                    PlotSpectrogram.plot(ds.getFeatures(), 2, 3);
//                }
//            }
log.info("Nrof eval files: " + eval.getNrofFiles() + " nrof examples: " + mdh.getNrofEvalExamples());
             //log.info(evaluation.stats());
            //log.info(evaluation.confusionToString());
            //log.info(roc.stats());
            //log.info(roc.stats());
            //mdh.fit();
            //mdh.resetTraining();
            //mdh.fit();
            //mdh.eval();
//            classifier = ModelSerializer.restoreMultiLayerNetwork(model2, false);
//            mdh = new GenericModelHandle(trainIter, evalIter, new MultiLayerModelAdapter(classifier), model, 1);
//            evaluation = mdh.createEvalTemplate();
//            roc = new ROCMultiClass();
//            mdh.eval(evaluation, roc);
//            log.info(evaluation.stats());
//            log.info(roc.stats());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}
