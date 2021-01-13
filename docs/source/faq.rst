F.A.Q.
=======
Here are a few questions that may perhaps come up.

- **Where can I download the training/testing data?**

    Check the :ref:`getting_started` section for those details.

- **What Python packages do I need?**

    Please see the requirements.txt or readme on Github.

- **Where in the code of the actual TM-Glow model?**

    The class you want to start at is :class:`nn.tmGlow.TMGlow`.

- **Why use an INN instead of a VAE or GANs?**

    VAEs have long suffered a lack of ability to yield crisp, complex images.
    GANs can be extremely unstable and were tested at the early stages of this work
    without success.
    Additionally, the INN has very nice probabilistic properties which is good for UQ.

- **My model is unstable during training!?!? Help!**

    Restart training at a checkpoint before your model diverged with :code:`--epoch-start #` parameter.
    Consider increasing gradient clipping, lower learning rate, increasing weight decay, etc. There are a lot
    of options to play with. While Glow is much more stable than GANs it can still be unstable at points.

- **Why are some of the sections of the program not fully documented?**

    Documenting takes time and energy. I'm working on slowly improving the docs, but it was my main focus to get
    the model and its components documented so people can understand what is going on under the hood.

- **How should I cite this?**

    Cite the paper!

    .. code-block:: text

        @article{geneva2020multi,
            title = "Multi-fidelity generative deep learning turbulent flows",
            author = "Nicholas Geneva and  Nicholas Zabaras",
            journal = "Foundations of Data Science",
            volume = "2",
            pages = "391",
            year = "2020",
            issn = "A0000-0002",
            doi = "10.3934/fods.2020019",
            url = "http://aimsciences.org/article/id/3a9f3d14-3421-4947-a45f-a9cc74edd097"
        }