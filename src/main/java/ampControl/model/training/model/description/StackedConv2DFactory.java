package ampControl.model.training.model.description;

import ampControl.model.training.data.iterators.CachingDataSetIterator;
import ampControl.model.training.model.*;
import ampControl.model.training.model.layerblocks.*;
import ampControl.model.training.model.layerblocks.graph.MinMaxPool;
import ampControl.model.training.model.layerblocks.graph.SeBlock;
import org.nd4j.linalg.activations.impl.ActivationReLU;
import org.nd4j.linalg.learning.config.Nesterovs;

import java.nio.file.Path;
import java.util.List;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * Description of a bunch of architectures which belong to a family of stacked 2D convolutional neural networks.
 *
 * @author Christian Skärby
 */
public class StackedConv2DFactory {
    private final CachingDataSetIterator trainIter;
    private final CachingDataSetIterator evalIter;
    private final int[] inputShape;
    private final String namePrefix;
    private final Path modelDir;

    public StackedConv2DFactory(CachingDataSetIterator trainIter, CachingDataSetIterator evalIter, int[] inputShape, String namePrefix, Path modelDir) {
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
//            final LayerBlockConfig zeroPad4x4 = new ZeroPad()
//                    .setPad_h_top(1)
//                    .setPad_h_bot(2)
//                    .setPad_w_left(1)
//                    .setPad_w_right(2);

        // Current best score with lgsc 96.0. Also performs very well in practice
        final LayerBlockConfig pool = new MinMaxPool().setSize(2).setStride(2);
        Stream.of(new IdBlock(), new SeBlock()).forEach(afterConvBlock ->
            IntStream.of(3).forEach(kernelSize ->
                DoubleStream.of(0).forEach(dropOutProb -> {
                    ModelBuilder builder = new DeserializingModelBuilder(modelDir.toString(),
                            new BlockBuilder()
                            .setStartingLearningRate(0.005)
                            .setUpdater(new Nesterovs(0.9))
                            .setNamePrefix(namePrefix)
                            .first(new ConvType(inputShape))
                            .andThenStack(2)
                            .of(new Conv2DBatchNormAfter()
                                    .setKernelSize(kernelSize)
                                    .setNrofKernels(64))
                            .andThen(afterConvBlock)
                            .andThen(pool)

                            .andThenStack(2)
                            .of(new Conv2DBatchNormAfter()
                                    .setKernelSize(kernelSize)
                                    .setNrofKernels(128))
                            .andThen(afterConvBlock)
                            .andThen(pool)

                            .andThenStack(2)
                            .of(new Conv2DBatchNormAfter()
                                    .setKernelSize(kernelSize)
                                    .setNrofKernels(256))
                            .andThen(afterConvBlock)
                            .andThen(pool)

                            .andThenStack(2)
                            .of(new Conv2DBatchNormAfter()
                                    .setKernelSize(kernelSize)
                                    .setNrofKernels(512))
                            .andThen(afterConvBlock)
                            .andThen(pool)

                            //.andThen(new GlobMeanMax())
                            .andThenStack(2)
                            .aggOf(new Dense()
                                    .setHiddenWidth(512)
                                    .setActivation(new ActivationReLU()))
                            .andFinally(new DropOut().setDropProb(dropOutProb))
                            .andFinally(new Output(trainIter.totalOutcomes())));
                    modelData.add(new GenericModelHandle(
                            trainIter,
                            evalIter,
                            new GraphModelAdapter(builder.buildGraph()),
                            builder.name(),
                            builder.getAccuracy()));
                })
            )
        );
    }
}
