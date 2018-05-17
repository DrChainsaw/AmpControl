package ampcontrol.model.training.model.layerblocks;

import ampcontrol.model.training.model.layerblocks.adapters.BuilderAdapter;

/**
 * Identity block. Does not add anything, just outputs the info. Useful for conditionals in architectures, e.g
 * {@code Stream.of(new IdBlock(), blockAlt1, blockAlt1).forEach(blockAlt -> createTheArchitectureWith(blockAlt))}
 */
public class IdBlock implements LayerBlockConfig {

    @Override
    public String name() {
        return "";
    }

    @Override
    public BlockInfo addLayers(BuilderAdapter builder, BlockInfo info) {
        return info;
    }
}
