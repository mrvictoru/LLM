###################################################################################################
# Entities and Relationships Extraction Prompt
###################################################################################################

graph_extraction_prompt ="""
Extract entities and relationships from the following text and format them for a knowledge graph. Focus primarily on types such as people, organisation or location used within the text. Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

- **Entity Types to Focus On:**
  - Person
  - Organization
  - Location
  - Tool
  - Concept
  - Process

-Steps-
1. Identify all distinct entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: Appropriate entity type
- entity_description: Comprehensive description of the entity's attributes and activities
2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
3. Return output in list of json of all the entities and relationships identified in steps 1 and 2, if no entities and relationships is identify, return an empty json.

JSON Format:
{extraction_json_formatting_prompt}

**Example 1:**
{extraction_example_1_prompt}

**Example 2:**
{extraction_example_2_prompt}

Now, based on these examples, please extract the distinct entities and relationships from the following text and return the information in the same structured format:

Here is the text:{text}

JSON OUPUT:
"""

extraction_json_formatting_prompt = """
{
  "entities": [
    {
      "entity_name": "Entity1",
      "entity_type": "Type1",
      "entity_description": "Description of Entity1's attributes and activities."
    },
    {
      "entity_name": "Entity2",
      "entity_type": "Type2",
      "entity_description": "Description of Entity2's attributes and activities."
    }
    // Add more entities as needed
  ],
  "relationships": [
    {
      "source_entity": "Entity1",
      "target_entity": "Entity2",
      "relationship_description": "Explanation of why Entity1 and Entity2 are related.",
      "relationship_strength": 0.9
    },
    {
      "source_entity": "Entity2",
      "target_entity": "Entity3",
      "relationship_description": "Explanation of why Entity2 and Entity3 are related.",
      "relationship_strength": 0.7
    }
    // Add more relationships as needed
  ]
}
"""

extraction_example_1_prompt = """
TEXT: "Alice, a software engineer at TechCorp, collaborated with Bob, a data scientist at DataWorks, on a machine learning project. The project aimed to improve the recommendation system for an e-commerce platform."

JSON OUPUT:
{
  "entities": [
    {
      "entity_name": "Alice",
      "entity_type": "Person",
      "entity_description": "A software engineer at TechCorp."
    },
    {
      "entity_name": "TechCorp",
      "entity_type": "Organization",
      "entity_description": "A technology company where Alice works."
    },
    {
      "entity_name": "Bob",
      "entity_type": "Person",
      "entity_description": "A data scientist at DataWorks."
    },
    {
      "entity_name": "DataWorks",
      "entity_type": "Organization",
      "entity_description": "A data science company where Bob works."
    },
    {
      "entity_name": "Machine Learning Project",
      "entity_type": "Project",
      "entity_description": "A project aimed to improve the recommendation system for an e-commerce platform."
    }
  ],
  "relationships": [
    {
      "source_entity": "Alice",
      "target_entity": "TechCorp",
      "relationship_description": "Alice works at TechCorp.",
      "relationship_strength": 0.9
    },
    {
      "source_entity": "Bob",
      "target_entity": "DataWorks",
      "relationship_description": "Bob works at DataWorks.",
      "relationship_strength": 0.9
    },
    {
      "source_entity": "Alice",
      "target_entity": "Bob",
      "relationship_description": "Alice collaborated with Bob on a machine learning project.",
      "relationship_strength": 0.8
    },
    {
      "source_entity": "Machine Learning Project",
      "target_entity": "Alice",
      "relationship_description": "Alice worked on the machine learning project.",
      "relationship_strength": 0.7
    },
    {
      "source_entity": "Machine Learning Project",
      "target_entity": "Bob",
      "relationship_description": "Bob worked on the machine learning project.",
      "relationship_strength": 0.7
    }
  ]
}
"""

extraction_example_2_prompt = """
TEXT: "Dr. Smith, a renowned cardiologist at HeartCare Hospital, published a research paper with Dr. Johnson, a neurologist at BrainHealth Institute, on the effects of stress on heart health."

JSON OUPUT: 
{
  "entities": [
    {
      "entity_name": "Dr. Smith",
      "entity_type": "Person",
      "entity_description": "A renowned cardiologist at HeartCare Hospital."
    },
    {
      "entity_name": "HeartCare Hospital",
      "entity_type": "Organization",
      "entity_description": "A hospital specializing in heart care where Dr. Smith works."
    },
    {
      "entity_name": "Dr. Johnson",
      "entity_type": "Person",
      "entity_description": "A neurologist at BrainHealth Institute."
    },
    {
      "entity_name": "BrainHealth Institute",
      "entity_type": "Organization",
      "entity_description": "An institute specializing in brain health where Dr. Johnson works."
    },
    {
      "entity_name": "Research Paper",
      "entity_type": "Publication",
      "entity_description": "A research paper on the effects of stress on heart health."
    }
  ],
  "relationships": [
    {
      "source_entity": "Dr. Smith",
      "target_entity": "HeartCare Hospital",
      "relationship_description": "Dr. Smith works at HeartCare Hospital.",
      "relationship_strength": 0.9
    },
    {
      "source_entity": "Dr. Johnson",
      "target_entity": "BrainHealth Institute",
      "relationship_description": "Dr. Johnson works at BrainHealth Institute.",
      "relationship_strength": 0.9
    },
    {
      "source_entity": "Dr. Smith",
      "target_entity": "Dr. Johnson",
      "relationship_description": "Dr. Smith published a research paper with Dr. Johnson.",
      "relationship_strength": 0.8
    },
    {
      "source_entity": "Research Paper",
      "target_entity": "Dr. Smith",
      "relationship_description": "Dr. Smith authored the research paper.",
      "relationship_strength": 0.7
    },
    {
      "source_entity": "Research Paper",
      "target_entity": "Dr. Johnson",
      "relationship_description": "Dr. Johnson authored the research paper.",
      "relationship_strength": 0.7
    }
  ]
}
"""

###################################################################################################
# Duplicate Entities Detection Prompt
###################################################################################################

check_duplicate_entities_prompt = """
Are the following two entities duplicates?

Entity 1:
Name: {entity1_name}
Type: {entity1_type}
Description: {entity1_description}

Entity 2:
Name: {entity2_name}
Type: {entity2_type}
Description: {entity2_description}

Answer 'yes' or 'no' only.
"""

summarize_descriptions_prompt = """
Summarize the following two descriptions into one coherent description:

Description 1: {description1}

Description 2: {description2}

Only provide the summary in the response.
Summary:
"""

###################################################################################################
# Community Report Generation Prompt
###################################################################################################

community_report_generation_prompt = """
You are an AI assistant that aids a human analyst in information discovery. This involves identifying and assessing relevant information related to entities (e.g., organizations, individuals) within a network.

# Goal
Generate a comprehensive community report based on a list of entities and their relationships, including optional claims. The report will inform decision-makers about the community's key entities, legal compliance, technical capabilities, reputation, and noteworthy claims.

# Report Structure
The report should include:
- **TITLE**: A short, specific name representing key entities, including representative named entities when possible.
- **SUMMARY**: An executive summary of the community's structure, entity relationships, and significant information.
- **IMPACT SEVERITY RATING**: A float score (0-10) indicating the community's impact severity.
- **RATING EXPLANATION**: A single sentence explaining the impact severity rating.
- **DETAILED FINDINGS**: 5-10 key insights about the community, each with a summary and explanatory text following the grounding rules below.

Return the output as a well-formed JSON-formatted string using the following format:
{community_report_format_prompt}

# Grounding Rules
- Support points with data references like:
  "Example sentence supported by data [Data: <dataset name> (record ids); <dataset name> (record ids)]."
- Limit to 5 record ids per reference, adding "+more" if necessary.
- Example:
  "Person X owns Company Y [Data: Reports (1), Entities (5,7); Relationships (23); Claims (7,2,34,64,46, +more)]."
- Exclude unsupported information.

# Example Input
-----------
{community_report_example_prompt}

# Real Data
Use the following Subgraph Data for your answer. Do not fabricate information.

Subgraph Data:
{input_text}

Output:
"""

community_report_format_prompt = """{
  "title": <report_title>,
  "summary": <executive_summary>,
  "rating": <impact_severity_rating>,
  "rating_explanation": <rating_explanation>,
  "findings": [
      {
          "summary":<insight_1_summary>,
          "explanation": <insight_1_explanation>
      },
      {
          "summary":<insight_2_summary>,
          "explanation": <insight_2_explanation>
      }
  ]
}"""

community_report_example_prompt = """
First example Subgraph Data:

Entities:
id,entity,description
1,ALICE,A software engineer at TechCorp.
2,TECHCORP,A technology company where Alice works.
3,BOB,A data scientist at DataWorks.
4,DATAWORKS,A data science company where Bob works.
5,MACHINE LEARNING PROJECT,A project aimed to improve the recommendation system for an e-commerce platform.

Relationships:
id,source,target,description
1,ALICE,TECHCORP,Alice works at TechCorp.
2,BOB,DATAWORKS,Bob works at DataWorks.
3,ALICE,BOB,Alice collaborated with Bob on a machine learning project.
4,MACHINE LEARNING PROJECT,ALICE,Alice worked on the machine learning project.
5,MACHINE LEARNING PROJECT,BOB,Bob worked on the machine learning project.

Output:
{
  "title": "Alice and Bob's Machine Learning Project",
  "summary": "The community revolves around Alice and Bob's collaboration on a machine learning project. Alice is a software engineer at TechCorp, and Bob is a data scientist at DataWorks. The project aims to improve the recommendation system for an e-commerce platform.",
  "rating": 7.0,
  "rating_explanation": "The impact severity rating is high due to the potential significant improvements in the e-commerce platform's recommendation system.",
  "findings": [
      {
          "summary": "Alice's role in the project",
          "explanation": "Alice, a software engineer at TechCorp, is a key contributor to the machine learning project. Her expertise in software engineering is crucial for the project's success. [Data: Entities (1), Relationships (1, 4)]"
      },
      {
          "summary": "Bob's role in the project",
          "explanation": "Bob, a data scientist at DataWorks, is another key contributor to the machine learning project. His expertise in data science is essential for the project's success. [Data: Entities (3), Relationships (2, 5)]"
      },
      {
          "summary": "Collaboration between Alice and Bob",
          "explanation": "Alice and Bob collaborated on the machine learning project, combining their expertise in software engineering and data science to improve the recommendation system. [Data: Relationships (3)]"
      },
      {
          "summary": "TechCorp's involvement",
          "explanation": "TechCorp, where Alice works, is indirectly involved in the project through Alice's contributions. The company's resources and support may have facilitated Alice's work on the project. [Data: Entities (2), Relationships (1)]"
      },
      {
          "summary": "DataWorks' involvement",
          "explanation": "DataWorks, where Bob works, is indirectly involved in the project through Bob's contributions. The company's resources and support may have facilitated Bob's work on the project. [Data: Entities (4), Relationships (2)]"
      }
  ]
}
Second example Subgraph Data:

Entities:
id,entity,description
1,LIBERTY HALL,Liberty Hall serves as a community center for various local events
2,PEACEMAKER SOCIETY,Peacemaker Society is dedicated to promoting harmony within the community

Relationships:
id,source,target,description

Output:
{
  "title": "Liberty Hall and Peacemaker Society",
  "summary": "The community includes Liberty Hall, a central location for local events, and the Peacemaker Society, an organization dedicated to promoting harmony. There are no direct relationships between the entities.",
  "rating": 3.0,
  "rating_explanation": "The impact severity rating is low as there are no existing relationships that could lead to potential conflicts or issues.",
  "findings": [
      {
          "summary": "Liberty Hall as a community center",
          "explanation": "Liberty Hall is a key entity in the community, serving as a hub for various local events. Its role as a central location fosters community engagement and participation. [Data: Entities (1)]"
      },
      {
          "summary": "Peacemaker Society's mission",
          "explanation": "Peacemaker Society is focused on promoting harmony within the community. Its efforts contribute to maintaining a peaceful and cooperative environment. [Data: Entities (2)]"
      }
  ]
}"""

###################################################################################################
# Vector Search Prompt
###################################################################################################

simple_query_answer_prompt = """
You are a helpful assistant that answer user's query about the content of a PDF document. Please generate an appropriate response based on the provided context from the pdf document and user's query. If the given context is not sufficient or not relevant, your response should be "I am sorry, I do not have the appropriate information to answer your question."
Here is the context for a given query:
{context}
Here is the user:
{query}
Response Output:
"""

###################################################################################################
# GraphRAG Map Global Search Prompt
###################################################################################################

map_global_search_prompt = """
Generate a response consisting of a list of key points that responds to the user's question, summarizing all relevant information in the input community report.

You should use the information provided in the community report below as the primary context for generating the response.
If you don't know the answer or if the input community report do not contain sufficient information to provide an answer, return and empty list. Do not make anything up.

Each key point in the response should have the following element:
- Description: A comprehensive description of the point.
- Importance Score: An integer score between 0-100 that indicates how important the point is in answering the user's question. An 'I don't know' type of response should have a score of 0.

The response should be JSON formatted as follows:
{map_response_format}

The response shall preserve the original meaning and use of modal verbs such as "shall", "may" or "will".

Points supported by data should list the relevant reports as references as follows:
"This is an example sentence supported by data references [Data: Reports (report ids)]"

**Do not list more than 5 record ids in a single reference**. Instead, list the top 5 most relevant record ids and add "+more" to indicate that there are more.

# Example Input:
{map_response_example_prompt}
The following are the input Community Report and User Query, generate appropriate response according to the json format. Do not include information where the supporting evidence for it is not provided.

---Community Report---
{context_data}
---User Query---
{user_query}
---Response Output---
"""

map_response_format_prompt = """
{
    "points": [
        {"description": "Description of point 1 [Data: Reports (report ids)]", "score": score_value},
        {"description": "Description of point 2 [Data: Reports (report ids)]", "score": score_value}
    ]
}
"""

map_response_example_prompt="""
---First Example Community Report---
{
  "title": "Alice and Bob's Machine Learning Project",
  "summary": "The community revolves around Alice and Bob's collaboration on a machine learning project. Alice is a software engineer at TechCorp, and Bob is a data scientist at DataWorks. The project aims to improve the recommendation system for an e-commerce platform.",
  "rating": 7.0,
  "rating_explanation": "The impact severity rating is high due to the potential significant improvements in the e-commerce platform's recommendation system.",
  "findings": [
      {
          "summary": "Alice's role in the project",
          "explanation": "Alice, a software engineer at TechCorp, is a key contributor to the machine learning project. Her expertise in software engineering is crucial for the project's success. [Data: Entities (1), Relationships (1, 4)]"
      },
      {
          "summary": "Bob's role in the project",
          "explanation": "Bob, a data scientist at DataWorks, is another key contributor to the machine learning project. His expertise in data science is essential for the project's success. [Data: Entities (3), Relationships (2, 5)]"
      },
      {
          "summary": "Collaboration between Alice and Bob",
          "explanation": "Alice and Bob collaborated on the machine learning project, combining their expertise in software engineering and data science to improve the recommendation system. [Data: Relationships (3)]"
      },
      {
          "summary": "TechCorp's involvement",
          "explanation": "TechCorp, where Alice works, is indirectly involved in the project through Alice's contributions. The company's resources and support may have facilitated Alice's work on the project. [Data: Entities (2), Relationships (1)]"
      },
      {
          "summary": "DataWorks' involvement",
          "explanation": "DataWorks, where Bob works, is indirectly involved in the project through Bob's contributions. The company's resources and support may have facilitated Bob's work on the project. [Data: Entities (4), Relationships (2)]"
      }
  ]
}
---User Query---
Which company is involved in the machine learning project?
---Response Output---
{
    "points": [
        {"description": "TechCorp's involvement in the project [Data: Reports (1)]", "score": 8.5},
        {"description": "DataWorks' role in supporting the project [Data: Reports (2)]", "score": 7.0}
    ]
}

---Second Example Community Report---
{
  "title": "Dr. Smith and Dr. Johnson's Research Collaboration",
  "summary": "The community revolves around Dr. Smith and Dr. Johnson's collaboration on a research paper about the effects of stress on heart health. Dr. Smith is a renowned cardiologist at HeartCare Hospital, and Dr. Johnson is a neurologist at BrainHealth Institute.",
  "rating": 8.0,
  "rating_explanation": "The impact severity rating is high due to the significant implications of their research on heart health.",
  "findings": [
      {
          "summary": "Dr. Smith's role in the research",
          "explanation": "Dr. Smith, a renowned cardiologist at HeartCare Hospital, is a key contributor to the research paper. His expertise in cardiology is crucial for the study's success. [Data: Entities (1), Relationships (1, 4)]"
      },
      {
          "summary": "Dr. Johnson's role in the research",
          "explanation": "Dr. Johnson, a neurologist at BrainHealth Institute, is another key contributor to the research paper. His expertise in neurology is essential for the study's success. [Data: Entities (3), Relationships (2, 5)]"
      },
      {
          "summary": "Collaboration between Dr. Smith and Dr. Johnson",
          "explanation": "Dr. Smith and Dr. Johnson collaborated on the research paper, combining their expertise in cardiology and neurology to study the effects of stress on heart health. [Data: Relationships (3)]"
      },
      {
          "summary": "HeartCare Hospital's involvement",
          "explanation": "HeartCare Hospital, where Dr. Smith works, is indirectly involved in the research through Dr. Smith's contributions. The hospital's resources and support may have facilitated Dr. Smith's work on the study. [Data: Entities (2), Relationships (1)]"
      },
      {
          "summary": "BrainHealth Institute's involvement",
          "explanation": "BrainHealth Institute, where Dr. Johnson works, is indirectly involved in the research through Dr. Johnson's contributions. The institute's resources and support may have facilitated Dr. Johnson's work on the study. [Data: Entities (4), Relationships (2)]"
      }
  ]
}
---User Query---
Who is Dr. Smith married to?
---Response Output---
{
    "points": []
}"""

reduce_global_search_prompt = """
Generate a response that answers the user's question by summarizing all the reports from multiple analysts who focused on different parts of the dataset. The response should match the target length and format.

The analysts' reports are ranked in descending order of importance.
If the answer is unknown or if the reports lack sufficient information, state this clearly without making anything up.
Exclude irrelevant information from the analysts' reports.
Merge relevant information into a comprehensive answer that explains all key points and implications, appropriate for the response length and format.
Add sections and commentary as appropriate for the length and format.
Style the response in Markdown.
Preserve the original meaning and use of modal verbs such as "shall", "may", or "will".
Preserve all data references included in the analysts' reports, but do not mention the roles of multiple analysts.
Do not list more than 5 record IDs in a single reference. Instead, list the top 5 most relevant record IDs and add "+more" to indicate additional records.

For example:
"Person X is the owner of Company Y and is subject to many allegations of wrongdoing [Data: Reports (2, 7, 34, 46, 64, +more)]. He is also the CEO of Company X [Data: Reports (1, 3)]."

where 1, 2, 3, 7, 34, 46, and 64 represent the IDs (not the indexes) of the relevant data records.

Do not include information without supporting evidence.

---Target Response Length and Format---
{response_type}
---Analyst Reports---
{report_data}
---User Query---
{user_query}
---Response Output---
"""

