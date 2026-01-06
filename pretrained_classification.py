{
  "cells": [
    {
      "cell_type": "markdown",
      "id": "MTYmjK3OEt4f",
      "metadata": {
        "id": "MTYmjK3OEt4f"
      },
      "source": [
        "# Description Model Approach 3: Fine-Tuning a Classification Model\n",
        "\n",
        "* Fine-tuned classification model. Use the csv files to fine-tune a pre-trained classification model. Apply the model to the linked-in data\n"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "NSUB4q_ht_Ql",
      "metadata": {
        "id": "NSUB4q_ht_Ql"
      },
      "source": []
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "bd3PW_CNDr2G",
      "metadata": {
        "id": "bd3PW_CNDr2G"
      },
      "outputs": [],
      "source": [
        "import pandas as pd"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "YXKX7JBtEcWa",
      "metadata": {
        "id": "YXKX7JBtEcWa"
      },
      "outputs": [],
      "source": [
        "GH_USER = \"luisadosch\"\n",
        "GH_REPO = \"Final-Project-snapAddy\"\n",
        "BRANCH = \"main\"\n",
        "\n",
        "\n",
        "def get_github_url(relative_path):\n",
        "    return f\"https://raw.githubusercontent.com/{GH_USER}/{GH_REPO}/{BRANCH}/{relative_path}\""
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "3IcFDwVJ5D5B",
      "metadata": {
        "id": "3IcFDwVJ5D5B"
      },
      "outputs": [],
      "source": [
        "jobs_annotated_csv_url = get_github_url(\"data/processed/jobs_annotated.csv\")\n",
        "jobs_annotated = pd.read_csv(jobs_annotated_csv_url)\n",
        "jobs_annotated.head()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "Iqt6nSUlIcPF",
      "metadata": {
        "id": "Iqt6nSUlIcPF"
      },
      "outputs": [],
      "source": [
        "len(\n",
        "    jobs_annotated[\n",
        "        (jobs_annotated[\"status\"] == \"ACTIVE\")\n",
        "    ]\n",
        ")\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "VqcmspCv-ED7",
      "metadata": {
        "id": "VqcmspCv-ED7"
      },
      "outputs": [],
      "source": [
        "seniority_url = get_github_url(\"data/raw/seniority-v2.csv\")\n",
        "df_seniority = pd.read_csv(seniority_url)\n",
        "df_seniority.head()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "Q_0xBj7pEk3D",
      "metadata": {
        "id": "Q_0xBj7pEk3D"
      },
      "outputs": [],
      "source": [
        "# from jobs_annotated only get the columns where status = ACTIVE\n",
        "df_seniority_test = jobs_annotated[(jobs_annotated[\"status\"] == \"ACTIVE\")]\n",
        "\n",
        "# now create df_seniority_test by only keeping position and renam eit in text, seniority, and cv_id\n",
        "df_seniority_test = df_seniority_test[[\"position\", \"seniority\", \"cv_id\"]].copy()\n",
        "\n",
        "\n",
        "# in df seniority_test, rename seniority in label and drop cv_id\n",
        "df_seniority_test.rename(columns={\"seniority\": \"label\"}, inplace=True)\n",
        "df_seniority_test.rename(columns={\"position\": \"text\"}, inplace=True)\n",
        "# drop column cv_id\n",
        "df_seniority_test.drop(columns=[\"cv_id\"], inplace=True)\n",
        "\n",
        "df_seniority_test.head()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "58gQ0bm8E3mC",
      "metadata": {
        "id": "58gQ0bm8E3mC"
      },
      "outputs": [],
      "source": [
        "df_seniority.head()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "xDb0YatcE_9K",
      "metadata": {
        "id": "xDb0YatcE_9K"
      },
      "outputs": [],
      "source": [
        "train_df = pd.DataFrame({\"text\": df_seniority[\"text\"], \"label\": df_seniority[\"label\"]})\n",
        "test_df = pd.DataFrame({\"text\": df_seniority_test[\"text\"], \"label\": df_seniority_test[\"label\"]})\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "CxZ0jJDAgSxp",
      "metadata": {
        "id": "CxZ0jJDAgSxp"
      },
      "outputs": [],
      "source": [
        "ord_map = {\n",
        "    \"Junior\": 1.0,\n",
        "    \"Professional\": 2.0,   # kommt im Train nicht vor, ist ok\n",
        "    \"Senior\": 3.0,\n",
        "    \"Lead\": 4.0,\n",
        "    \"Management\": 5.0,\n",
        "    \"Director\": 6.0\n",
        "}\n",
        "\n",
        "train_df[\"y_reg\"] = train_df[\"label\"].map(ord_map).astype(float)\n",
        "test_df[\"y_reg\"]  = test_df[\"label\"].map(ord_map).astype(float)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "RswMZEEFHCds",
      "metadata": {
        "id": "RswMZEEFHCds"
      },
      "outputs": [],
      "source": [
        "# len of train_df\n",
        "len(train_df)\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "2NrWYkkVHFnT",
      "metadata": {
        "id": "2NrWYkkVHFnT"
      },
      "outputs": [],
      "source": [
        "len(test_df)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "4DeTq6ozLwQw",
      "metadata": {
        "id": "4DeTq6ozLwQw"
      },
      "outputs": [],
      "source": [
        "from transformers import AutoTokenizer\n",
        "\n",
        "model_ckpt = \"xlm-roberta-base\"\n",
        "tokenizer = AutoTokenizer.from_pretrained(model_ckpt)\n",
        "\n",
        "MAX_LEN = 48  # Jobtitel sind kurz; wenn viele sehr lang sind: 48\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "fuYMyj1bpcSA",
      "metadata": {
        "id": "fuYMyj1bpcSA"
      },
      "outputs": [],
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "from datasets import Dataset\n",
        "\n",
        "train_sub_df, val_df = train_test_split(\n",
        "    train_df,\n",
        "    test_size=0.2,\n",
        "    stratify=train_df[\"label\"],  # oder stratify=train_df[\"y_reg\"].astype(int)\n",
        "    random_state=42\n",
        ")\n",
        "\n",
        "train_ds = Dataset.from_pandas(\n",
        "    train_sub_df[[\"text\",\"y_reg\"]].rename(columns={\"y_reg\":\"labels\"}).copy(),\n",
        "    preserve_index=False\n",
        ")\n",
        "\n",
        "val_ds = Dataset.from_pandas(\n",
        "    val_df[[\"text\",\"y_reg\"]].rename(columns={\"y_reg\":\"labels\"}).copy(),\n",
        "    preserve_index=False\n",
        ")\n",
        "\n",
        "# dein finaler Test bleibt separat:\n",
        "test_ds = Dataset.from_pandas(\n",
        "    test_df[[\"text\",\"y_reg\"]].rename(columns={\"y_reg\":\"labels\"}).copy(),\n",
        "    preserve_index=False\n",
        ")\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "ToI2Q-sSX7x_",
      "metadata": {
        "id": "ToI2Q-sSX7x_"
      },
      "outputs": [],
      "source": [
        "def tokenize(batch):\n",
        "    return tokenizer(\n",
        "        batch[\"text\"],\n",
        "        truncation=True,\n",
        "        padding=\"max_length\",\n",
        "        max_length=MAX_LEN\n",
        "    )\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 15,
      "id": "3QDYuxQfps0h",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 113,
          "referenced_widgets": [
            "b8894245dc8346b0845fed271f0d7a48",
            "c6547ed287684bfbab9d053b47cb9c30",
            "eba7d10dc66540b4b4bfdfbb0ca3407b",
            "cdffaa3cdca94512834c89b4a65aaaa6",
            "c9389465ebea480e818b34a555f65705",
            "52b1b29bd639436ebb4d18b7c00c243b",
            "3f7415e44e6149deb0c8ea8ecea8d8dd",
            "62e9905f373342d2a3f2c76162659a61",
            "aa07d6d8f55f4da7b2c633ae3b40304e",
            "8cf1a3bab8784c5083d3171677d4dac5",
            "7c77f327cb6343d2a6f17d6156c779fc",
            "2d7c8d50c06f4a1db5840fe7e5944dc6",
            "4295dfedcba44bce8892dd77ba8ec93f",
            "4c9b95eac22347ed92f91428ba6ad27c",
            "9d7e38ae90244ceab9c874ef963c8f7d",
            "89d2e1be54124cf6b43851fe837cb17d",
            "97ce0fa84b6a4e828bcf746e46f2a0f2",
            "845dc76433cf42dba4a66320d28ed779",
            "a53fb4a069d4428a865e512dd94dd001",
            "7e1a5d78ff1c42f182ddc783508941e2",
            "a6af92fad39b482a90c39f9885f2748d",
            "6baed3df444642fab854bdb3db8f4331",
            "1df7121ec1a744cb82c30eeb4be56e1c",
            "aefa22dbfe0b45a8a6ae02b6378157dd",
            "3bbc9eebd016411281a2789686f97665",
            "944b9049d32a437a89c70f05534bddaf",
            "4684b4ae61144955bd59f90ab22cc9ce",
            "cf29b805795a4708ab77ec641ba0b28f",
            "0cbf519985aa488fa43fbd0464aa5198",
            "d8cfe3b0f22540d5b5fcc46bdc93879b",
            "91cb58f781dd4aa3a6c4697836a8b5dd",
            "dd5c63d2d94348ebaf5f86856a837c33",
            "e25d11c77a7846cbb52825a6e14b75ed"
          ]
        },
        "id": "3QDYuxQfps0h",
        "outputId": "a44f9f3a-48c7-487f-9f6f-1058d941e446"
      },
      "outputs": [
        {
          "data": {
            "application/vnd.jupyter.widget-view+json": {
              "model_id": "b8894245dc8346b0845fed271f0d7a48",
              "version_major": 2,
              "version_minor": 0
            },
            "text/plain": [
              "Map:   0%|          | 0/7542 [00:00<?, ? examples/s]"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "application/vnd.jupyter.widget-view+json": {
              "model_id": "2d7c8d50c06f4a1db5840fe7e5944dc6",
              "version_major": 2,
              "version_minor": 0
            },
            "text/plain": [
              "Map:   0%|          | 0/1886 [00:00<?, ? examples/s]"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "application/vnd.jupyter.widget-view+json": {
              "model_id": "1df7121ec1a744cb82c30eeb4be56e1c",
              "version_major": 2,
              "version_minor": 0
            },
            "text/plain": [
              "Map:   0%|          | 0/623 [00:00<?, ? examples/s]"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "train_ds = train_ds.map(tokenize, batched=True)\n",
        "val_ds   = val_ds.map(tokenize, batched=True)\n",
        "test_ds  = test_ds.map(tokenize, batched=True)\n",
        "\n",
        "cols = [\"input_ids\", \"attention_mask\", \"labels\"]\n",
        "train_ds.set_format(type=\"torch\", columns=cols)\n",
        "val_ds.set_format(type=\"torch\", columns=cols)\n",
        "test_ds.set_format(type=\"torch\", columns=cols)\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 16,
      "id": "DwfmgtW2M1QS",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 121,
          "referenced_widgets": [
            "09de3025727f4fd580325b7059163573",
            "5ff190891f1445b89e13585039e50987",
            "9eb99a9c0e9d4dd3a12bef3765789e04",
            "5240aac51ae34713a4dd5a2a4176a4ba",
            "c81f0abe7a104b60b4d2d3e1a214b8f2",
            "aafd88fc3eb04c53aa65988ac11b4263",
            "931722e2a4be495dbce4306cd41e8ddb",
            "632684c1ceab4d2dbc104a79a2ee5e72",
            "77769c67032443b9b99165d43c7b7cde",
            "ced4a089a9ab4311b1bef0abe22c7ed9",
            "32d77e26f6804a76800728726d5dcc53"
          ]
        },
        "id": "DwfmgtW2M1QS",
        "outputId": "760060a5-beca-4492-dd1b-1fc0d9775dcf"
      },
      "outputs": [
        {
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "WARNING:torchao.kernel.intmm:Warning: Detected no triton, on systems without Triton certain kernels will not work\n"
          ]
        },
        {
          "data": {
            "application/vnd.jupyter.widget-view+json": {
              "model_id": "09de3025727f4fd580325b7059163573",
              "version_major": 2,
              "version_minor": 0
            },
            "text/plain": [
              "model.safetensors:   0%|          | 0.00/1.12G [00:00<?, ?B/s]"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "Some weights of XLMRobertaForSequenceClassification were not initialized from the model checkpoint at xlm-roberta-base and are newly initialized: ['classifier.dense.bias', 'classifier.dense.weight', 'classifier.out_proj.bias', 'classifier.out_proj.weight']\n",
            "You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.\n"
          ]
        }
      ],
      "source": [
        "from transformers import AutoModelForSequenceClassification\n",
        "\n",
        "\n",
        "model = AutoModelForSequenceClassification.from_pretrained(\n",
        "    model_ckpt,\n",
        "    num_labels=1,\n",
        "    problem_type=\"regression\"\n",
        ")\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 17,
      "id": "XJwrv881iqLk",
      "metadata": {
        "id": "XJwrv881iqLk"
      },
      "outputs": [],
      "source": [
        "import numpy as np\n",
        "from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error\n",
        "\n",
        "def score_to_label(s):\n",
        "    if s < 1.5: return \"Junior\"\n",
        "    if s < 2.5: return \"Professional\"\n",
        "    if s < 3.5: return \"Senior\"\n",
        "    if s < 4.5: return \"Lead\"\n",
        "    if s < 5.5: return \"Management\"\n",
        "    return \"Director\"\n",
        "\n",
        "def compute_metrics_reg(eval_pred):\n",
        "    preds, labels = eval_pred\n",
        "    scores = np.squeeze(preds)     # kontinuierliche Vorhersage\n",
        "    mae = mean_absolute_error(labels, scores)\n",
        "\n",
        "    y_pred = [score_to_label(s) for s in scores]\n",
        "    # labels sind 1..6 floats -> zurück zu Text\n",
        "    inv_ord = {v:k for k,v in ord_map.items()}\n",
        "    y_true = [inv_ord[float(int(round(x)))] for x in labels]\n",
        "\n",
        "    acc = accuracy_score(y_true, y_pred)\n",
        "    f1m = f1_score(y_true, y_pred, average=\"macro\")\n",
        "    return {\"mae\": mae, \"acc_thresh\": acc, \"f1_macro\": f1m}\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 18,
      "id": "IjjQVBdjPPtQ",
      "metadata": {
        "id": "IjjQVBdjPPtQ"
      },
      "outputs": [],
      "source": [
        "import os\n",
        "os.environ[\"WANDB_DISABLED\"] = \"true\"\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 19,
      "id": "mZCyfqHKNOop",
      "metadata": {
        "id": "mZCyfqHKNOop"
      },
      "outputs": [],
      "source": [
        "from transformers import TrainingArguments\n",
        "from transformers import EarlyStoppingCallback\n",
        "\n",
        "\n",
        "args = TrainingArguments(\n",
        "    output_dir=\"seniority_ft\",\n",
        "    learning_rate=1e-5,\n",
        "    per_device_train_batch_size=16,\n",
        "    per_device_eval_batch_size=32,\n",
        "    num_train_epochs=10,\n",
        "    weight_decay=0.05,\n",
        "    warmup_ratio=0.06,\n",
        "    eval_strategy=\"epoch\",\n",
        "    save_strategy=\"epoch\",\n",
        "    load_best_model_at_end=True,\n",
        "    metric_for_best_model=\"mae\",\n",
        "    greater_is_better=False,\n",
        "    fp16=True,\n",
        "    logging_steps=50,\n",
        "    report_to=\"none\",              # wichtig\n",
        "    dataloader_num_workers=0       # oft stabiler in Colab\n",
        ")\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "gSTPpkjfN7bz",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 223
        },
        "id": "gSTPpkjfN7bz",
        "outputId": "9cb5cce4-af3b-4619-f428-43e9394f5062"
      },
      "outputs": [
        {
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "/tmp/ipython-input-3360937659.py:3: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `Trainer.__init__`. Use `processing_class` instead.\n",
            "  trainer = Trainer(\n",
            "/usr/local/lib/python3.12/dist-packages/torch/utils/data/dataloader.py:668: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.\n",
            "  warnings.warn(warn_msg)\n"
          ]
        },
        {
          "data": {
            "text/html": [
              "\n",
              "    <div>\n",
              "      \n",
              "      <progress value='21' max='4720' style='width:300px; height:20px; vertical-align: middle;'></progress>\n",
              "      [  21/4720 02:29 < 10:16:13, 0.13 it/s, Epoch 0.04/10]\n",
              "    </div>\n",
              "    <table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              " <tr style=\"text-align: left;\">\n",
              "      <th>Epoch</th>\n",
              "      <th>Training Loss</th>\n",
              "      <th>Validation Loss</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "  </tbody>\n",
              "</table><p>"
            ],
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        },
        {
          "data": {
            "text/html": [
              "\n",
              "    <div>\n",
              "      \n",
              "      <progress value='26' max='4720' style='width:300px; height:20px; vertical-align: middle;'></progress>\n",
              "      [  26/4720 03:09 < 10:18:07, 0.13 it/s, Epoch 0.05/10]\n",
              "    </div>\n",
              "    <table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              " <tr style=\"text-align: left;\">\n",
              "      <th>Epoch</th>\n",
              "      <th>Training Loss</th>\n",
              "      <th>Validation Loss</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "  </tbody>\n",
              "</table><p>"
            ],
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "from transformers import Trainer\n",
        "\n",
        "trainer = Trainer(\n",
        "    model=model,\n",
        "    args=args,\n",
        "    train_dataset=train_ds,\n",
        "    eval_dataset=val_ds,\n",
        "    tokenizer=tokenizer,\n",
        "    compute_metrics=compute_metrics_reg,\n",
        "    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]\n",
        ")\n",
        "\n",
        "trainer.train()\n",
        "\n",
        "trainer.evaluate()\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "sFSeq2oCq_-I",
      "metadata": {
        "id": "sFSeq2oCq_-I"
      },
      "outputs": [],
      "source": [
        "trainer.evaluate(test_ds)"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "nAUNkPOXyOgO",
      "metadata": {
        "id": "nAUNkPOXyOgO"
      },
      "source": [
        "So basically the model works really well with trian but not with test\n",
        "- when looking which labels performed worst its professional -> so the model is bad at predicting that because it is not part of the train df\n",
        "\n",
        "-> so that is why we need more trianing data but without data leakage"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "0ov7Km9kW3ld",
      "metadata": {
        "id": "0ov7Km9kW3ld"
      },
      "source": [
        "Ty model with new synthethic data"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "24YsajefqbGO",
      "metadata": {
        "id": "24YsajefqbGO"
      },
      "outputs": [],
      "source": [
        "synthetic_url = get_github_url(\"data/results/gemini_synthetic.csv\")\n",
        "data_synthetic = pd.read_csv(synthetic_url )\n",
        "data_synthetic.head()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "tg8tK4xbaabj",
      "metadata": {
        "id": "tg8tK4xbaabj"
      },
      "outputs": [],
      "source": [
        "# change data_synthetic to only keep columns position, seniority\n",
        "\n",
        "data_synthetic = data_synthetic[[\"position\", \"seniority\"]].copy()\n",
        "\n"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "6jw40NDueSyY",
      "metadata": {
        "id": "6jw40NDueSyY"
      },
      "source": [
        "2. Train with synthetic data"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "6Trh35uvd_Cb",
      "metadata": {
        "id": "6Trh35uvd_Cb"
      },
      "outputs": [],
      "source": [
        "# dein ursprüngliches Mapping\n",
        "ord_map = {\n",
        "    \"Junior\": 1.0,\n",
        "    \"Professional\": 2.0,\n",
        "    \"Senior\": 3.0,\n",
        "    \"Lead\": 4.0,\n",
        "    \"Management\": 5.0,\n",
        "    \"Director\": 6.0\n",
        "}\n",
        "\n",
        "# Mapping umdrehen: Zahl -> String\n",
        "id2label = {v: k for k, v in ord_map.items()}\n",
        "\n",
        "# neue Label-Spalte erzeugen\n",
        "data_synthetic[\"label\"] = data_synthetic[\"seniority\"].map(id2label)\n",
        "\n",
        "# rename column position into text\n",
        "data_synthetic.rename(columns={\"position\": \"text\"}, inplace=True)\n",
        "# rename seniority into \ty_reg\n",
        "data_synthetic.rename(columns={\"seniority\": \"y_reg\"}, inplace=True)\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "RZYOjJWxeZI-",
      "metadata": {
        "id": "RZYOjJWxeZI-"
      },
      "outputs": [],
      "source": [
        "data_synthetic.head()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "5uX8ex-je93g",
      "metadata": {
        "id": "5uX8ex-je93g"
      },
      "outputs": [],
      "source": [
        "# Anzahl NaNs pro Spalte\n",
        "data_synthetic.isna().sum()\n",
        "\n",
        "\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "d3-prDLUfjP_",
      "metadata": {
        "id": "d3-prDLUfjP_"
      },
      "outputs": [],
      "source": [
        "# Zeilen mit NaN in y_reg oder label entfernen\n",
        "data_synthetic = data_synthetic.dropna(subset=[\"y_reg\", \"label\"])\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "6vSO5rICfluO",
      "metadata": {
        "id": "6vSO5rICfluO"
      },
      "outputs": [],
      "source": [
        "data_synthetic[[\"y_reg\", \"label\"]].isna().sum()\n",
        "len(data_synthetic)\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "AcHpTlfPf6lr",
      "metadata": {
        "id": "AcHpTlfPf6lr"
      },
      "outputs": [],
      "source": [
        "data_synthetic"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "wgH_B2y2eyrp",
      "metadata": {
        "id": "wgH_B2y2eyrp"
      },
      "outputs": [],
      "source": [
        "# keep only the columns you want to train on\n",
        "syn_rows = data_synthetic[[\"text\", \"label\"]].copy()          # add \"y_reg\" too if you use regression\n",
        "# syn_rows = data_synthetic[[\"text\", \"label\", \"y_reg\"]].copy()\n",
        "\n",
        "train_rows = train_df[[\"text\", \"label\"]].copy()\n",
        "# train_rows = train_df[[\"text\", \"label\", \"y_reg\"]].copy()\n",
        "\n",
        "# append rows\n",
        "train_df_aug = pd.concat([train_rows, syn_rows], ignore_index=True)\n",
        "\n",
        "print(len(train_df), \"->\", len(train_df_aug))\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "4Dvog2PcgM-b",
      "metadata": {
        "id": "4Dvog2PcgM-b"
      },
      "outputs": [],
      "source": [
        "train_df_aug.head()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "Zpje6f0TgwO3",
      "metadata": {
        "id": "Zpje6f0TgwO3"
      },
      "outputs": [],
      "source": [
        "label_list = sorted(train_df_aug[\"label\"].unique())\n",
        "label_list"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "4Ma1w5_WhLB7",
      "metadata": {
        "id": "4Ma1w5_WhLB7"
      },
      "outputs": [],
      "source": [
        "train_df_aug[\"label\"].value_counts()\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "9G6xbc2Egjro",
      "metadata": {
        "id": "9G6xbc2Egjro"
      },
      "outputs": [],
      "source": [
        "from sklearn.model_selection import train_test_split\n",
        "\n",
        "train_sub_df, val_df = train_test_split(\n",
        "    train_df_aug,\n",
        "    test_size=0.2,\n",
        "    stratify=train_df_aug[\"label\"],\n",
        "    random_state=42\n",
        ")\n",
        "\n",
        "print(\"Train:\\n\", train_sub_df[\"label\"].value_counts())\n",
        "print(\"Val:\\n\", val_df[\"label\"].value_counts())\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "DWIk3UcQgsIw",
      "metadata": {
        "id": "DWIk3UcQgsIw"
      },
      "outputs": [],
      "source": [
        "label_list = sorted(train_sub_df[\"label\"].unique())\n",
        "label2id = {l: i for i, l in enumerate(label_list)}\n",
        "id2label = {i: l for l, i in label2id.items()}\n",
        "\n",
        "print(label_list)\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "tQWIYKqFiT8r",
      "metadata": {
        "id": "tQWIYKqFiT8r"
      },
      "outputs": [],
      "source": [
        "train_sub_df[\"label_id\"] = train_sub_df[\"label\"].map(label2id)\n",
        "val_df[\"label_id\"] = val_df[\"label\"].map(label2id)\n",
        "test_df[\"label_id\"] = test_df[\"label\"].map(label2id)\n",
        "\n",
        "print(\"test missing:\", test_df[\"label_id\"].isna().sum())\n",
        "print(\"train missing:\", train_sub_df[\"label_id\"].isna().sum())\n",
        "print(\"val missing:\", val_df[\"label_id\"].isna().sum())\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "4KJIzN_gijr1",
      "metadata": {
        "id": "4KJIzN_gijr1"
      },
      "outputs": [],
      "source": [
        "test_df"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "eNmg2EwXiz-k",
      "metadata": {
        "id": "eNmg2EwXiz-k"
      },
      "outputs": [],
      "source": [
        "from datasets import Dataset\n",
        "\n",
        "train_hf_df = train_sub_df[[\"text\", \"label_id\"]].rename(columns={\"label_id\": \"labels\"}).copy()\n",
        "val_hf_df   = val_df[[\"text\", \"label_id\"]].rename(columns={\"label_id\": \"labels\"}).copy()\n",
        "test_hf_df  = test_df[[\"text\", \"label_id\"]].rename(columns={\"label_id\": \"labels\"}).copy()\n",
        "\n",
        "train_ds = Dataset.from_pandas(train_hf_df, preserve_index=False)\n",
        "val_ds   = Dataset.from_pandas(val_hf_df, preserve_index=False)\n",
        "test_ds  = Dataset.from_pandas(test_hf_df, preserve_index=False)\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "cVvVEnQLjobj",
      "metadata": {
        "id": "cVvVEnQLjobj"
      },
      "outputs": [],
      "source": [
        "from transformers import AutoTokenizer\n",
        "\n",
        "model_ckpt = \"xlm-roberta-base\"\n",
        "tokenizer = AutoTokenizer.from_pretrained(model_ckpt)\n",
        "\n",
        "MAX_LEN = 48  # Jobtitel sind kurz; wenn viele sehr lang sind: 48"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "URQOX2jrjvVd",
      "metadata": {
        "id": "URQOX2jrjvVd"
      },
      "outputs": [],
      "source": [
        "train_ds = train_ds.map(tokenize, batched=True)\n",
        "val_ds   = val_ds.map(tokenize, batched=True)\n",
        "test_ds  = test_ds.map(tokenize, batched=True)\n",
        "\n",
        "cols = [\"input_ids\", \"attention_mask\", \"labels\"]\n",
        "train_ds.set_format(type=\"torch\", columns=cols)\n",
        "val_ds.set_format(type=\"torch\", columns=cols)\n",
        "test_ds.set_format(type=\"torch\", columns=cols)\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "LAbhxOVGkViD",
      "metadata": {
        "id": "LAbhxOVGkViD"
      },
      "outputs": [],
      "source": [
        "from transformers import AutoModelForSequenceClassification\n",
        "\n",
        "model = AutoModelForSequenceClassification.from_pretrained(\n",
        "    model_ckpt,\n",
        "    num_labels=len(label2id),\n",
        "    id2label=id2label,\n",
        "    label2id=label2id\n",
        ")\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "lfe219pTkXup",
      "metadata": {
        "id": "lfe219pTkXup"
      },
      "outputs": [],
      "source": [
        "import numpy as np\n",
        "from sklearn.metrics import accuracy_score, f1_score\n",
        "\n",
        "def compute_metrics(eval_pred):\n",
        "    logits, labels = eval_pred\n",
        "    preds = np.argmax(logits, axis=-1)\n",
        "    return {\n",
        "        \"accuracy\": accuracy_score(labels, preds),\n",
        "        \"f1_macro\": f1_score(labels, preds, average=\"macro\")\n",
        "    }\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "R4wmfgLhkbei",
      "metadata": {
        "id": "R4wmfgLhkbei"
      },
      "outputs": [],
      "source": [
        "from transformers import TrainingArguments\n",
        "\n",
        "args = TrainingArguments(\n",
        "    output_dir=\"seniority_cls_xlmr\",\n",
        "    learning_rate=1e-5,\n",
        "    per_device_train_batch_size=16,\n",
        "    per_device_eval_batch_size=32,\n",
        "    num_train_epochs=10,\n",
        "    weight_decay=0.05,\n",
        "    warmup_ratio=0.06,\n",
        "    eval_strategy=\"epoch\",\n",
        "    save_strategy=\"epoch\",\n",
        "    load_best_model_at_end=True,\n",
        "    metric_for_best_model=\"f1_macro\",\n",
        "    greater_is_better=True,\n",
        "    fp16=True,\n",
        "    logging_steps=50,\n",
        "    report_to=\"none\",\n",
        "    dataloader_num_workers=0\n",
        ")\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "PQ0L5NzVkdti",
      "metadata": {
        "id": "PQ0L5NzVkdti"
      },
      "outputs": [],
      "source": [
        "from transformers import Trainer, EarlyStoppingCallback\n",
        "\n",
        "trainer = Trainer(\n",
        "    model=model,\n",
        "    args=args,\n",
        "    train_dataset=train_ds,\n",
        "    eval_dataset=val_ds,\n",
        "    tokenizer=tokenizer,\n",
        "    compute_metrics=compute_metrics,\n",
        "    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]\n",
        ")\n",
        "\n",
        "trainer.train()\n",
        "trainer.evaluate()\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "GnocwAYWkhHd",
      "metadata": {
        "id": "GnocwAYWkhHd"
      },
      "outputs": [],
      "source": [
        "trainer.evaluate(test_ds)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "Ym2RlqDHm84Y",
      "metadata": {
        "id": "Ym2RlqDHm84Y"
      },
      "outputs": [],
      "source": [
        "from sklearn.metrics import classification_report\n",
        "import numpy as np\n",
        "\n",
        "pred = trainer.predict(val_ds)\n",
        "y_pred = np.argmax(pred.predictions, axis=-1)\n",
        "y_true = pred.label_ids\n",
        "\n",
        "print(classification_report(\n",
        "    y_true, y_pred,\n",
        "    target_names=[id2label[i] for i in range(len(id2label))]\n",
        "))\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "1WxcuhjqnCc6",
      "metadata": {
        "id": "1WxcuhjqnCc6"
      },
      "outputs": [],
      "source": [
        "import numpy as np\n",
        "\n",
        "counts = train_sub_df[\"label\"].value_counts()\n",
        "weights = np.zeros(len(label2id), dtype=np.float32)\n",
        "\n",
        "for lbl, c in counts.items():\n",
        "    weights[label2id[lbl]] = 1.0 / c\n",
        "\n",
        "weights = weights / weights.mean()  # normalisieren\n",
        "weights\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "DPqGYNxNnGvQ",
      "metadata": {
        "id": "DPqGYNxNnGvQ"
      },
      "outputs": [],
      "source": [
        "import torch\n",
        "from transformers import Trainer\n",
        "import torch.nn as nn\n",
        "\n",
        "class WeightedTrainer(Trainer):\n",
        "    def __init__(self, class_weights=None, *args, **kwargs):\n",
        "        super().__init__(*args, **kwargs)\n",
        "        self.class_weights = class_weights\n",
        "\n",
        "    def compute_loss(self, model, inputs, return_outputs=False):\n",
        "        labels = inputs.get(\"labels\")\n",
        "        outputs = model(**{k:v for k,v in inputs.items() if k != \"labels\"})\n",
        "        logits = outputs.logits\n",
        "\n",
        "        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))\n",
        "        loss = loss_fct(logits, labels)\n",
        "\n",
        "        return (loss, outputs) if return_outputs else loss\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "vpejnnj2nJQQ",
      "metadata": {
        "id": "vpejnnj2nJQQ"
      },
      "outputs": [],
      "source": [
        "class_weights = torch.tensor(weights)\n",
        "\n",
        "trainer = WeightedTrainer(\n",
        "    model=model,\n",
        "    args=args,\n",
        "    train_dataset=train_ds,\n",
        "    eval_dataset=val_ds,\n",
        "    tokenizer=tokenizer,\n",
        "    compute_metrics=compute_metrics,\n",
        "    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],\n",
        "    class_weights=class_weights\n",
        ")\n",
        "\n",
        "trainer.train()\n",
        "trainer.evaluate()\n"
      ]
    },
    {
      "cell_type": "markdown",
      "id": "MOaTVf8RoYTj",
      "metadata": {
        "id": "MOaTVf8RoYTj"
      },
      "source": [
        "orsample"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "EevM4KR-nLxY",
      "metadata": {
        "id": "EevM4KR-nLxY"
      },
      "outputs": [],
      "source": [
        "import pandas as pd\n",
        "\n",
        "prof_df = train_sub_df[train_sub_df[\"label\"] == \"Professional\"]\n",
        "rest_df = train_sub_df[train_sub_df[\"label\"] != \"Professional\"]\n",
        "\n",
        "# z.B. Professional 5x wiederholen\n",
        "train_sub_df_up = pd.concat([rest_df, prof_df.sample(len(prof_df)*5, replace=True, random_state=42)], ignore_index=True)\n",
        "\n",
        "train_sub_df_up[\"label_id\"] = train_sub_df_up[\"label\"].map(label2id)\n"
      ]
    }
  ],
  "metadata": {
    "accelerator": "GPU",
    "colab": {
      "gpuType": "T4",
      "provenance": []
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.12.12"
    },
    "widgets": {
      "application/vnd.jupyter.widget-state+json": {
        "09de3025727f4fd580325b7059163573": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "HBoxModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HBoxModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HBoxView",
            "box_style": "",
            "children": [
              "IPY_MODEL_5ff190891f1445b89e13585039e50987",
              "IPY_MODEL_9eb99a9c0e9d4dd3a12bef3765789e04",
              "IPY_MODEL_5240aac51ae34713a4dd5a2a4176a4ba"
            ],
            "layout": "IPY_MODEL_c81f0abe7a104b60b4d2d3e1a214b8f2"
          }
        },
        "0cbf519985aa488fa43fbd0464aa5198": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "DescriptionStyleModel",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "DescriptionStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "description_width": ""
          }
        },
        "1df7121ec1a744cb82c30eeb4be56e1c": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "HBoxModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HBoxModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HBoxView",
            "box_style": "",
            "children": [
              "IPY_MODEL_aefa22dbfe0b45a8a6ae02b6378157dd",
              "IPY_MODEL_3bbc9eebd016411281a2789686f97665",
              "IPY_MODEL_944b9049d32a437a89c70f05534bddaf"
            ],
            "layout": "IPY_MODEL_4684b4ae61144955bd59f90ab22cc9ce"
          }
        },
        "2d7c8d50c06f4a1db5840fe7e5944dc6": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "HBoxModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HBoxModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HBoxView",
            "box_style": "",
            "children": [
              "IPY_MODEL_4295dfedcba44bce8892dd77ba8ec93f",
              "IPY_MODEL_4c9b95eac22347ed92f91428ba6ad27c",
              "IPY_MODEL_9d7e38ae90244ceab9c874ef963c8f7d"
            ],
            "layout": "IPY_MODEL_89d2e1be54124cf6b43851fe837cb17d"
          }
        },
        "32d77e26f6804a76800728726d5dcc53": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "DescriptionStyleModel",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "DescriptionStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "description_width": ""
          }
        },
        "3bbc9eebd016411281a2789686f97665": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "FloatProgressModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "FloatProgressModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "ProgressView",
            "bar_style": "success",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_d8cfe3b0f22540d5b5fcc46bdc93879b",
            "max": 623,
            "min": 0,
            "orientation": "horizontal",
            "style": "IPY_MODEL_91cb58f781dd4aa3a6c4697836a8b5dd",
            "value": 623
          }
        },
        "3f7415e44e6149deb0c8ea8ecea8d8dd": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "DescriptionStyleModel",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "DescriptionStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "description_width": ""
          }
        },
        "4295dfedcba44bce8892dd77ba8ec93f": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "HTMLModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HTMLModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HTMLView",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_97ce0fa84b6a4e828bcf746e46f2a0f2",
            "placeholder": "​",
            "style": "IPY_MODEL_845dc76433cf42dba4a66320d28ed779",
            "value": "Map: 100%"
          }
        },
        "4684b4ae61144955bd59f90ab22cc9ce": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "4c9b95eac22347ed92f91428ba6ad27c": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "FloatProgressModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "FloatProgressModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "ProgressView",
            "bar_style": "success",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_a53fb4a069d4428a865e512dd94dd001",
            "max": 1886,
            "min": 0,
            "orientation": "horizontal",
            "style": "IPY_MODEL_7e1a5d78ff1c42f182ddc783508941e2",
            "value": 1886
          }
        },
        "5240aac51ae34713a4dd5a2a4176a4ba": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "HTMLModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HTMLModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HTMLView",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_ced4a089a9ab4311b1bef0abe22c7ed9",
            "placeholder": "​",
            "style": "IPY_MODEL_32d77e26f6804a76800728726d5dcc53",
            "value": " 1.12G/1.12G [00:09&lt;00:00, 200MB/s]"
          }
        },
        "52b1b29bd639436ebb4d18b7c00c243b": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "5ff190891f1445b89e13585039e50987": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "HTMLModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HTMLModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HTMLView",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_aafd88fc3eb04c53aa65988ac11b4263",
            "placeholder": "​",
            "style": "IPY_MODEL_931722e2a4be495dbce4306cd41e8ddb",
            "value": "model.safetensors: 100%"
          }
        },
        "62e9905f373342d2a3f2c76162659a61": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "632684c1ceab4d2dbc104a79a2ee5e72": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "6baed3df444642fab854bdb3db8f4331": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "DescriptionStyleModel",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "DescriptionStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "description_width": ""
          }
        },
        "77769c67032443b9b99165d43c7b7cde": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "ProgressStyleModel",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "ProgressStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "bar_color": null,
            "description_width": ""
          }
        },
        "7c77f327cb6343d2a6f17d6156c779fc": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "DescriptionStyleModel",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "DescriptionStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "description_width": ""
          }
        },
        "7e1a5d78ff1c42f182ddc783508941e2": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "ProgressStyleModel",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "ProgressStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "bar_color": null,
            "description_width": ""
          }
        },
        "845dc76433cf42dba4a66320d28ed779": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "DescriptionStyleModel",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "DescriptionStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "description_width": ""
          }
        },
        "89d2e1be54124cf6b43851fe837cb17d": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "8cf1a3bab8784c5083d3171677d4dac5": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "91cb58f781dd4aa3a6c4697836a8b5dd": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "ProgressStyleModel",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "ProgressStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "bar_color": null,
            "description_width": ""
          }
        },
        "931722e2a4be495dbce4306cd41e8ddb": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "DescriptionStyleModel",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "DescriptionStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "description_width": ""
          }
        },
        "944b9049d32a437a89c70f05534bddaf": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "HTMLModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HTMLModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HTMLView",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_dd5c63d2d94348ebaf5f86856a837c33",
            "placeholder": "​",
            "style": "IPY_MODEL_e25d11c77a7846cbb52825a6e14b75ed",
            "value": " 623/623 [00:00&lt;00:00, 6258.45 examples/s]"
          }
        },
        "97ce0fa84b6a4e828bcf746e46f2a0f2": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "9d7e38ae90244ceab9c874ef963c8f7d": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "HTMLModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HTMLModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HTMLView",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_a6af92fad39b482a90c39f9885f2748d",
            "placeholder": "​",
            "style": "IPY_MODEL_6baed3df444642fab854bdb3db8f4331",
            "value": " 1886/1886 [00:00&lt;00:00, 9701.60 examples/s]"
          }
        },
        "9eb99a9c0e9d4dd3a12bef3765789e04": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "FloatProgressModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "FloatProgressModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "ProgressView",
            "bar_style": "success",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_632684c1ceab4d2dbc104a79a2ee5e72",
            "max": 1115567652,
            "min": 0,
            "orientation": "horizontal",
            "style": "IPY_MODEL_77769c67032443b9b99165d43c7b7cde",
            "value": 1115567652
          }
        },
        "a53fb4a069d4428a865e512dd94dd001": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "a6af92fad39b482a90c39f9885f2748d": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "aa07d6d8f55f4da7b2c633ae3b40304e": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "ProgressStyleModel",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "ProgressStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "bar_color": null,
            "description_width": ""
          }
        },
        "aafd88fc3eb04c53aa65988ac11b4263": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "aefa22dbfe0b45a8a6ae02b6378157dd": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "HTMLModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HTMLModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HTMLView",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_cf29b805795a4708ab77ec641ba0b28f",
            "placeholder": "​",
            "style": "IPY_MODEL_0cbf519985aa488fa43fbd0464aa5198",
            "value": "Map: 100%"
          }
        },
        "b8894245dc8346b0845fed271f0d7a48": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "HBoxModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HBoxModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HBoxView",
            "box_style": "",
            "children": [
              "IPY_MODEL_c6547ed287684bfbab9d053b47cb9c30",
              "IPY_MODEL_eba7d10dc66540b4b4bfdfbb0ca3407b",
              "IPY_MODEL_cdffaa3cdca94512834c89b4a65aaaa6"
            ],
            "layout": "IPY_MODEL_c9389465ebea480e818b34a555f65705"
          }
        },
        "c6547ed287684bfbab9d053b47cb9c30": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "HTMLModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HTMLModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HTMLView",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_52b1b29bd639436ebb4d18b7c00c243b",
            "placeholder": "​",
            "style": "IPY_MODEL_3f7415e44e6149deb0c8ea8ecea8d8dd",
            "value": "Map: 100%"
          }
        },
        "c81f0abe7a104b60b4d2d3e1a214b8f2": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "c9389465ebea480e818b34a555f65705": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "cdffaa3cdca94512834c89b4a65aaaa6": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "HTMLModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "HTMLModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "HTMLView",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_8cf1a3bab8784c5083d3171677d4dac5",
            "placeholder": "​",
            "style": "IPY_MODEL_7c77f327cb6343d2a6f17d6156c779fc",
            "value": " 7542/7542 [00:00&lt;00:00, 13164.25 examples/s]"
          }
        },
        "ced4a089a9ab4311b1bef0abe22c7ed9": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "cf29b805795a4708ab77ec641ba0b28f": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "d8cfe3b0f22540d5b5fcc46bdc93879b": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "dd5c63d2d94348ebaf5f86856a837c33": {
          "model_module": "@jupyter-widgets/base",
          "model_module_version": "1.2.0",
          "model_name": "LayoutModel",
          "state": {
            "_model_module": "@jupyter-widgets/base",
            "_model_module_version": "1.2.0",
            "_model_name": "LayoutModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "LayoutView",
            "align_content": null,
            "align_items": null,
            "align_self": null,
            "border": null,
            "bottom": null,
            "display": null,
            "flex": null,
            "flex_flow": null,
            "grid_area": null,
            "grid_auto_columns": null,
            "grid_auto_flow": null,
            "grid_auto_rows": null,
            "grid_column": null,
            "grid_gap": null,
            "grid_row": null,
            "grid_template_areas": null,
            "grid_template_columns": null,
            "grid_template_rows": null,
            "height": null,
            "justify_content": null,
            "justify_items": null,
            "left": null,
            "margin": null,
            "max_height": null,
            "max_width": null,
            "min_height": null,
            "min_width": null,
            "object_fit": null,
            "object_position": null,
            "order": null,
            "overflow": null,
            "overflow_x": null,
            "overflow_y": null,
            "padding": null,
            "right": null,
            "top": null,
            "visibility": null,
            "width": null
          }
        },
        "e25d11c77a7846cbb52825a6e14b75ed": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "DescriptionStyleModel",
          "state": {
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "DescriptionStyleModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/base",
            "_view_module_version": "1.2.0",
            "_view_name": "StyleView",
            "description_width": ""
          }
        },
        "eba7d10dc66540b4b4bfdfbb0ca3407b": {
          "model_module": "@jupyter-widgets/controls",
          "model_module_version": "1.5.0",
          "model_name": "FloatProgressModel",
          "state": {
            "_dom_classes": [],
            "_model_module": "@jupyter-widgets/controls",
            "_model_module_version": "1.5.0",
            "_model_name": "FloatProgressModel",
            "_view_count": null,
            "_view_module": "@jupyter-widgets/controls",
            "_view_module_version": "1.5.0",
            "_view_name": "ProgressView",
            "bar_style": "success",
            "description": "",
            "description_tooltip": null,
            "layout": "IPY_MODEL_62e9905f373342d2a3f2c76162659a61",
            "max": 7542,
            "min": 0,
            "orientation": "horizontal",
            "style": "IPY_MODEL_aa07d6d8f55f4da7b2c633ae3b40304e",
            "value": 7542
          }
        }
      }
    }
  },
  "nbformat": 4,
  "nbformat_minor": 5
}
