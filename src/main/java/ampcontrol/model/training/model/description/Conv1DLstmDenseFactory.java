package ampcontrol.model.training.model.description;

import ampcontrol.model.training.data.iterators.MiniEpochDataSetIterator;
import ampcontrol.model.training.data.iterators.preprocs.CnnToRnnToLastStepToFfPreProcessor;
import ampcontrol.model.training.model.GenericModelHandle;
import ampcontrol.model.training.model.GraphModelAdapter;
import ampcontrol.model.training.model.ModelAdapterWithPreProc;
import ampcontrol.model.training.model.ModelHandle;
import ampcontrol.model.training.model.builder.BlockBuilder;
import ampcontrol.model.training.model.builder.DeserializingModelBuilder;
import ampcontrol.model.training.model.builder.ModelBuilder;
import ampcontrol.model.training.model.layerblocks.*;
import ampcontrol.model.training.model.layerblocks.graph.GlobMeanMax;
import ampcontrol.model.training.model.naming.FileNamePolicy;
import org.nd4j.linalg.activations.impl.ActivationIdentity;
import org.nd4j.linalg.activations.impl.ActivationReLU;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.schedule.ScheduleType;
import org.nd4j.linalg.schedule.StepSchedule;

import java.util.List;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

/**
 * Description of a bunch of architectures which belong to a family of 1D convolutional neural networks followed by LSTM
 * which in turn are followed by fully connected layers.
 *
 * @author Christian Skärby
 */
public class Conv1DLstmDenseFactory {
    private final MiniEpochDataSetIterator trainIter;
    private final MiniEpochDataSetIterator evalIter;
    private final int[] inputShape;
    private final String namePrefix;
    private final FileNamePolicy modelFileNamePolicy;

    public Conv1DLstmDenseFactory(MiniEpochDataSetIterator trainIter, MiniEpochDataSetIterator evalIter, int[] inputShape, String namePrefix, FileNamePolicy modelFileNamePolicy) {
        this.trainIter = trainIter;
        this.evalIter = evalIter;
        this.inputShape = inputShape;
        this.namePrefix = namePrefix;
        this.modelFileNamePolicy = modelFileNamePolicy;
    }

    /**
     * Adds the ModelHandles defined by this class to the given list
     *
     * @param modelData list to add models to
     */
    public void addModelData(List<ModelHandle> modelData) {
        final DataSetPreProcessor rnnPreproc = new CnnToRnnToLastStepToFfPreProcessor();
// About 92% (maybe room for more) after 30k iterations without dropout. 3 best, then 1 then 7 then 0. 7 Seems to catch up slowly...
// final Supplier<ProcessingResult.Factory> audioPostProcessingSupplier = () -> new Pipe(
//                new Spectrogrammm(512, 32, 1),
//                new UnitStdZeroMean()
//        );

        // DONT FORGET BATCH SIZE 64 i 2x faster compared to 32

        IntStream.of(1).forEach(nrofCnnLayers -> {


            DoubleStream.of(0).forEach(dropOutProb -> {
                ModelBuilder builder = new DeserializingModelBuilder(modelFileNamePolicy,
                        new BlockBuilder()
                                .setNamePrefix(namePrefix)
                                .setUpdater(new Adam(new StepSchedule(ScheduleType.ITERATION, 0.05, 0.1, 40000)))
                                .first(new RnnType(inputShape))
                                .andThen(new Conv1D()
                                        .setNrofKernels(256))
                                .andThenStack(nrofCnnLayers)
                                .res()
                                .aggOf(new Conv1D()
                                        .setNrofKernels(256))
                                .andThen(new Conv1D()
                                        .setNrofKernels(256)
                                        .setActivation(new ActivationIdentity()))
                                .andFinally(new ZeroPad1D().setPaddingLeft(3).setPaddingRight(3))
                                // .andFinally(new DropOut().setDropProb(dropOutProb/8))
                                //.andThen(new LstmBlock().setWidth(128))
                                //.andThenStack(1)
                                //.res()
                                // .of(new LstmBlock().setWidth(128))
                                //.andFinally(new LstmBlock().setWidth(128)
                                //       .setActivation(new ActivationIdentity()))
                                //.andFinally(new Norm())
                                //.andFinally(new DropOut().setDropProb(dropOutProb / 4))
                                //   .andThen(new LastStep())
                                .andThen(new GlobMeanMax())
                                .andThenStack(2)
                                //.res()
                                .aggOf(new Dense()
                                        .setHiddenWidth(256)
                                        .setActivation(new ActivationReLU()))
                                .andFinally(new DropOut().setDropProb(dropOutProb))
                                .andFinally(new Output(trainIter.totalOutcomes())));

                modelData.add(new GenericModelHandle(
                        trainIter,
                        evalIter,
                        new ModelAdapterWithPreProc(rnnPreproc,
                                new GraphModelAdapter(builder.buildGraph())),
                        builder.name()));
            });
        });
    }
}
