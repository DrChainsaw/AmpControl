package ampcontrol.model.training;

import ampcontrol.model.training.listen.IterationSupplier;
import ampcontrol.model.training.listen.TimeMeasurement;
import ampcontrol.model.training.listen.TrainEvaluator;
import ampcontrol.model.training.listen.TrainScoreListener;
import ampcontrol.model.training.model.ModelHandle;
import ampcontrol.model.training.model.naming.FileNamePolicy;
import ampcontrol.model.training.model.validation.*;
import ampcontrol.model.training.model.validation.listen.*;
import ampcontrol.model.visualize.Plot;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.time.Duration;
import java.time.LocalTime;
import java.util.List;
import java.util.function.Consumer;
import java.util.function.Predicate;
import java.util.function.Supplier;

/**
 * Harness for training and evaluation of {@link ModelHandle ModelHandles}. Models will take turn in doing fitting. Will
 * evaluate models in regular intervals and saves the last evaluated model as long as the accuracy is not worse than 90%
 * of the best evaluation accuracy for the model. Will also plot accuracy and score for training, eval and best eval.
 *
 * @author Christian Skärby
 */
class TrainingHarness {

    private static final Logger log = LoggerFactory.getLogger(TrainingHarness.class);

    private static final boolean doStatsLogging = false;
    private static final int evalEveryNrofSteps = 100;
    private static final int nrofStepsBeforeFirstEval = 200;
    private static final String bestSuffix = "_best";
    private static final String scoreSuffix = ".score";
    private static final double saveThreshold = 0.9;

    private static final String trainEvalPrefix = "Train";
    private static final String lastEvalPrefix = "LastEval";
    private static final String bestEvalPrefix = "BestEval";

    private final List<ModelHandle> modelsToTrain;
    private final FileNamePolicy baseFileNamePolicy;
    private final Plot.Factory<Integer, Double> plotFactory;
    private final TextWriter.Factory writerFactory;

    TrainingHarness(
            List<ModelHandle> modelsToTrain,
            FileNamePolicy baseFileNamePolicy,
            Plot.Factory<Integer, Double> plotFactory,
            TextWriter.Factory writerFactory) {
        this.modelsToTrain = modelsToTrain;
        this.baseFileNamePolicy = baseFileNamePolicy;
        this.plotFactory = plotFactory;
        this.writerFactory = writerFactory;
    }

    private final class EvalValidationFactory implements Validation.Factory<Evaluation> {

        private final ModelHandle model;
        private final Plot<Integer, Double> evalPlot;
        private final Plot<Integer, Double> scorePlot;
        private final String fileBaseName;

        private EvalValidationFactory(ModelHandle model,
                                      Plot<Integer, Double> evalPlot,
                                      Plot<Integer, Double> scorePlot) {
            this.model = model;
            this.evalPlot = evalPlot;
            this.scorePlot = scorePlot;
            this.fileBaseName = baseFileNamePolicy.toFileName(model.name());
        }

        @Override
        public Validation<Evaluation> create(List<String> labels) {

            try {
                final BestEvalScore bestEvalScore = new BestEvalScore(fileBaseName + bestSuffix + scoreSuffix);
                log.info("Accuracy for model " + model.name() + ": " + bestEvalScore.get());
                IterationSupplier iterListener = new IterationSupplier();
                model.getModel().addListeners(iterListener);

                final Consumer<Evaluation> listener =
                        createEvalConsumer(bestEvalScore, iterListener)
                                .andThen(createLastCheckPoint(bestEvalScore))
                                .andThen(createBestCheckPoint(bestEvalScore, iterListener))
                                .andThen(bestEvalScore);

                return decorate(new EvalValidation(new Evaluation(labels), listener), bestEvalScore);

            } catch (IOException e) {
                throw new IllegalArgumentException("Could not load file for model " + model.name() + "!", e);
            }
        }

        private Consumer<Evaluation> createEvalConsumer(final Supplier<Double> bestEvalSupplier, final Supplier<Integer> iterationSupplier) {
            final String lastEvalLabel = lastEvalName(model.name());
            final Consumer<Evaluation> plotEval = eval -> evalPlot.plotData(lastEvalLabel, iterationSupplier.get(), eval.accuracy());
            final Consumer<Evaluation> plotScore = eval -> scorePlot.plotData(lastEvalLabel, iterationSupplier.get(), model.getModel().score());

            final Consumer<Evaluation> storePlots =
                    eval -> {
                        try {
                            evalPlot.storePlotData(trainEvalName(model.name()));
                            evalPlot.storePlotData(lastEvalLabel);
                            scorePlot.storePlotData(trainEvalName(model.name()));
                            scorePlot.storePlotData(lastEvalLabel);
                        } catch (IOException e) {
                            log.warn("Failed to store plots! {}", e);
                        }
                    };

            final EvalLog evalLog = new EvalLog(model.name(), bestEvalSupplier);

            return new NewThread<>( // Background work
                    new Synced<>( // To avoid mixed up logging
                            this,
                            evalLog
                                    .andThen(plotEval)
                                    .andThen(plotScore)
                                    .andThen(storePlots)));
        }

        private Consumer<Evaluation> createLastCheckPoint(final Supplier<Double> bestEvalSupplier) {
            final Consumer<Evaluation> saveCheckPoint = createCheckPoint(fileBaseName);
            
            final Predicate<Evaluation> gate = eval -> eval.accuracy() >= bestEvalSupplier.get() * saveThreshold;

            return new NewThread<>( // Background work
                    new Gated<>(saveCheckPoint, gate));
        }

        private Consumer<Evaluation> createBestCheckPoint(final Supplier<Double> bestEvalSupplier,
                                                          final Supplier<Integer> iterationSupplier) {
            final Consumer<Evaluation> saveCheckPoint = createCheckPoint(fileBaseName + bestSuffix);

            final String bestEvalLabel = bestEvalName(model.name());
            final Consumer<Evaluation> plotEval = eval -> evalPlot.plotData(bestEvalLabel, iterationSupplier.get(), eval.accuracy());
            final Consumer<Evaluation> plotScore = eval -> scorePlot.plotData(bestEvalLabel, iterationSupplier.get(), model.getModel().score());

            final Consumer<Evaluation> storePlots =
                    eval -> {
                        try {
                            evalPlot.storePlotData(bestEvalLabel);
                            scorePlot.storePlotData(bestEvalLabel);
                        } catch (IOException e) {
                            log.warn("Failed to store plots! {}", e);
                        }
                    };

            final Predicate<Evaluation> gate = eval -> eval.accuracy() >= bestEvalSupplier.get();

            return new NewThread<>( // Background work
                    new Gated<>(saveCheckPoint
                            .andThen(plotEval)
                            .andThen(plotScore)
                            .andThen(storePlots), gate));
        }

        private Consumer<Evaluation> createCheckPoint(final String fileBaseName) {
            final Consumer<Evaluation> scoreCheckPoint = new EvalCheckPoint(fileBaseName + scoreSuffix, model.name(), writerFactory);
            final Consumer<Evaluation> modelCheckPointEc = eval -> {
                try {
                    model.saveModel(fileBaseName);
                } catch (IOException e) {
                    log.warn("Failed to store model {}! {}", model.name(), e);
                }
            };
            return modelCheckPointEc.andThen(scoreCheckPoint);
        }

        private Validation<Evaluation> decorate(final Validation<Evaluation> evaluationValidation, final Supplier<Double> bestEvalSupplier) {
            final Consumer<Boolean> logEval = willEval -> {
                if (willEval) {
                    log.info("Begin eval of " + model.name());
                }
            };
            final Consumer<Boolean> logAccuracy = willEval -> log.info("Current best " + bestEvalSupplier.get() + " for model: " + model.name());
            return new TimeMeasuring(
                    new Listening<>(logAccuracy.andThen(logEval),
                            new Skipping<>(eval -> evalEveryNrofSteps, nrofStepsBeforeFirstEval,
                                    new Skipping<>(eval -> (int) Math.floor(10 * (1 - eval.accuracy())), "Skip eval: ", // TODO: Break out and test?
                                            evaluationValidation
                                    )
                            )
                    )
            );
        }
    }

    private void addListeners(final List<ModelHandle> models) {

        for (final ModelHandle mh : models) {
            if (doStatsLogging) {
                final UIServer uiServer = UIServer.getInstance();
                //Alternative: new FileStatsStorage(File) - see UIStorageExample
                final StatsStorage statsStorage = new FileStatsStorage(new File(mh.name() + "_stats"));
                uiServer.attach(statsStorage);
                mh.getModel().addListeners(new StatsListener(statsStorage, 20));
            }
            mh.getModel().addListeners(new TimeMeasurement());
            mh.getModel().addListeners(new TrainScoreListener((i, s) -> log.info("Score at iter " + i + ": " + s)));
            // mh.getModel().addListeners(new SeBlockInspection());
        }
    }

    private void addValidation(final List<ModelHandle> models) {
        final Plot<Integer, Double> evalPlot = initPlot("Accuracy", models);
        final Plot<Integer, Double> scorePlot = initPlot("Score", models);

        for (ModelHandle mh : models) {
            final String trainName = trainEvalName(mh.name());
            mh.getModel().addListeners(new TrainScoreListener((i, s) -> scorePlot.plotData(trainName, i, s)));
            mh.getModel().addListeners(new TrainEvaluator((i, e) -> evalPlot.plotData(trainName, i, e)));
            mh.registerValidation(new EvalValidationFactory(mh, evalPlot, scorePlot));
        }
    }

    private Plot<Integer, Double> initPlot(final String title, final List<ModelHandle> models) {
        final Plot<Integer, Double> plot = plotFactory.create(title);
        for (ModelHandle md : models) {
            plot.createSeries(trainEvalName(md.name()));
            plot.createSeries(lastEvalName(md.name()));
            plot.createSeries(bestEvalName(md.name()));
        }
        return plot;
    }

    /**
     * Trains the models
     *
     * @param maxNrofTrainingSteps number of training steps
     */
    void startTraining(final int maxNrofTrainingSteps) {

        addListeners(modelsToTrain);
        addValidation(modelsToTrain);

        Nd4j.getMemoryManager().setAutoGcWindow(5000);
        for (int trainingStep = 0; trainingStep < maxNrofTrainingSteps; trainingStep++) {
            log.info("****************************** Training step " + trainingStep + " started! ***************************************");
            final LocalTime beforeTrain = LocalTime.now();
            for (ModelHandle mh : modelsToTrain) {
                log.info("Training model: " + mh.name());
                mh.fit();
            }
            final long totalTrainingTime = Duration.between(beforeTrain, LocalTime.now()).toMillis();
            log.info("Total training time: " + totalTrainingTime);

            for (ModelHandle mh : modelsToTrain) {
                mh.eval();
                mh.resetTraining();
            }
        }
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

}
