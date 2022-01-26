from typing import Optional, Sequence

import wandb

class WandbLogger:

    def __init__(self,
                 project: Optional[str] = None,
                 entity: Optional[str] = None,
                 config: Optional[dict] = None,
                 name: Optional[str] = None):
        """
        Args:
            project: project name.
            entity: user/team name.
            config: run's configuration.
            name: name of the run.
        """
        self.run = wandb.init(
            project=project,
            entity=entity,
            config=config,
            name=name,
        )

    def log(self, data: dict):
        self.instance.log(data)
