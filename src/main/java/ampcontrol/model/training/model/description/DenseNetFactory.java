package ampcontrol.model.training.model.description;

import ampcontrol.model.training.data.iterators.CachingDataSetIterator;
import ampcontrol.model.training.model.*;
import ampcontrol.model.training.model.layerblocks.*;
import ampcontrol.model.training.model.layerblocks.graph.MinMaxPool;
import ampcontrol.model.training.model.layerblocks.graph.SeBlock;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.schedule.ScheduleType;
import org.nd4j.linalg.schedule.StepSchedule;

import java.nio.file.Path;
import java.util.List;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

/**
 * Description of a bunch of architectures which belong to a family of dense nets.
 * <br><br>
 * See https://arxiv.org/abs/1608.06993
 *
 * @author Christian Skärby
 */
public class DenseNetFactory {
    private final CachingDataSetIterator trainIter;
    private final CachingDataSetIterator evalIter;
    private final int[] inputShape;
    private final String namePrefix;
    private final Path modelDir;

    public DenseNetFactory(CachingDataSetIterator trainIter, CachingDataSetIterator evalIter, int[] inputShape, String namePrefix, Path modelDir) {
        this.trainIter = trainIter;
        this.evalIter = evalIter;
        this.inputShape = inputShape;
        this.namePrefix = namePrefix;
        this.modelDir = modelDir;
    }

    /**
     * Adds the ModelHandles defined by this class to the given list
     *
     * @param modelData list to add models to
     */
    public void addModelData(List<ModelHandle> modelData) {

        final LayerBlockConfig zeroPad3x3 = new ZeroPad().setPad(1);
        IntStream.of(4, 8, 16).forEach(denseStackSize -> {
            DoubleStream.of(0).forEach(dropOutProb -> {
                // ~0.96 acc, but quite heavy
                ModelBuilder builder = new DeserializingModelBuilder(modelDir.toString(),
                        new BlockBuilder()
                                .setNamePrefix(namePrefix)
                                .setUpdater(new Nesterovs(new StepSchedule(ScheduleType.ITERATION, 0.001, 0.1, 40000)))
                                .first(new ConvType(inputShape))
                                .andThen(zeroPad3x3)
                                .andThen(new Conv2DBatchNormAfter()
                                        .setKernelSize(3)
                                        .setNrofKernels(64))
                                .andThen(new MinMaxPool().setSize(3).setStride(3))
                                .andThen(zeroPad3x3)
                                .andThen(new Conv2DBatchNormAfter()
                                        .setKernelSize(3)
                                        .setNrofKernels(128))
                                .andThen(new MinMaxPool().setSize(3).setStride(3))
                                .andThen(new SeBlock())
                                .andThen(zeroPad3x3)
                                .andThen(new Conv2DBatchNormAfter()
                                        .setKernelSize(3)
                                        .setNrofKernels(128))
                                .andThen(new MinMaxPool().setSize(3).setStride(3))
                                .andThen(new SeBlock())
                                .andThenStack(10)
                                .aggDenseStack(denseStackSize)
                                .aggOf(zeroPad3x3)
                                .andFinally(new Conv2DBatchNormAfter()
                                        .setKernelSize(3)
                                        .setNrofKernels(32))
                                .andThen(new Conv2DBatchNormAfter()
                                        .setNrofKernels(32 * 4)
                                        .setKernelSize(1))
                                //   .andThen(new DropOut().setDropProb(dropOutProb))
                                .andFinally(new SeBlock())
                                .andThenStack(2)
                                .of(new Dense())//.setActivation(new ActivationSELU()))
                                .andThen(new DropOut().setDropProb(dropOutProb))
                                .andFinally(new Output(trainIter.totalOutcomes())));
                modelData.add(new GenericModelHandle(
                        trainIter,
                        evalIter,
                        new GraphModelAdapter(builder.buildGraph()),
                        builder.name()));
            });
        });
    }
}
