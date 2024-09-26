// mdStore.ts
import { create } from "zustand";
import axios from "axios";
import { updateMarkdownContent, UpdateMarkdownRequest } from "@/api/extract"; // 确保路径正确

interface MdContent {
  content: string;
  isLoading: boolean;
}

type AnchorType =
  | "span"
  | "div"
  | "comment"
  | "data-attribute"
  | "hr"
  | "mark"
  | "p";

interface AnchorOptions {
  type: AnchorType;
  prefix?: string;
  style?: string;
  className?: string;
  customAttributes?: Record<string, string>;
}

const defaultAnchorOptions: AnchorOptions = {
  type: "span",
  prefix: "md-anchor-",
  style: "display:none;",
  className: "",
  customAttributes: {},
};

interface MdState {
  mdContents: Record<string, MdContent>;
  allMdContent: string;
  allMdContentWithAnchor: string;
  error: Error | null;
  currentRequestId: number;
  setMdUrlArr: (urls: string[]) => Promise<void>;
  getAllMdContent: (data: string[]) => string;
  setAllMdContent: (val?: string) => void;
  setAllMdContentWithAnchor: (val?: string) => void;
  getContentWithAnchors: (
    data: string[],
    options?: Partial<AnchorOptions>
  ) => string;
  jumpToAnchor: (anchorId: string) => number;
  reset: () => void;
  updateMdContent: (
    fileKey: string,
    pageNumber: string | number,
    newContent: string
  ) => Promise<void>;
}

const MAX_CONCURRENT_REQUESTS = 2;

const initialState = {
  mdContents: {},
  allMdContent: "",
  allMdContentWithAnchor: "",
  error: null,
  currentRequestId: 0,
};

const useMdStore = create<MdState>((set, get) => ({
  ...initialState,

  reset: () => {
    set(initialState);
  },

  setAllMdContent: (value?: string) => {
    set(() => ({
      allMdContent: value,
    }));
  },

  setAllMdContentWithAnchor: (value?: string) => {
    set(() => ({
      allMdContentWithAnchor: value,
    }));
  },

  setMdUrlArr: async (urls: string[]) => {
    const requestId = get().currentRequestId + 1;
    set((state) => ({ currentRequestId: requestId, error: null }));

    const fetchContent = async (url: string): Promise<[string, string]> => {
      try {
        const response = await axios.get<string>(url);
        return [url, response.data];
      } catch (error) {
        if (get().currentRequestId === requestId) {
          set((state) => ({ error: error as Error }));
        }
        return [url, ""];
      }
    };

    const fetchWithConcurrency = async (
      urls: string[]
    ): Promise<[string, string][]> => {
      const queue = [...urls];
      const results: [string, string][] = [];
      const inProgress = new Set<Promise<[string, string]>>();

      while (queue.length > 0 || inProgress.size > 0) {
        while (inProgress.size < MAX_CONCURRENT_REQUESTS && queue.length > 0) {
          const url = queue.shift()!;
          const promise = fetchContent(url);
          inProgress.add(promise);
          promise.then((result) => {
            results.push(result);
            inProgress.delete(promise);
          });
        }
        if (inProgress.size > 0) {
          await Promise.race(inProgress);
        }
      }

      return results;
    };

    const results = await fetchWithConcurrency(urls);

    if (get().currentRequestId === requestId) {
      const newMdContents: Record<string, MdContent> = {};
      results.forEach(([url, content]) => {
        newMdContents[url] = { content, isLoading: false };
      });

      set((state) => ({
        mdContents: newMdContents,
        allMdContent: state.getAllMdContent(results.map((i) => i[1])),
        allMdContentWithAnchor: state.getContentWithAnchors(
          results.map((i) => i[1])
        ),
      }));
    }
  },

  getAllMdContent: (data) => {
    return data?.join("\n\n");
  },

  getContentWithAnchors: (data: string[], options?: Partial<AnchorOptions>) => {
    const opts = { ...defaultAnchorOptions, ...options };

    const generateAnchorTag = (index: number) => {
      const id = `${opts.prefix}${index}`;
      const attributes = Object.entries(opts.customAttributes || {})
        .map(([key, value]) => `${key}="${value}"`)
        .join(" ");

      switch (opts.type) {
        case "span":
        case "div":
        case "mark":
        case "p":
          return `<${opts.type} id="${id}" style="${opts.style}" class="${opts.className}" ${attributes}></${opts.type}>`;
        case "comment":
          return `<!-- anchor: ${id} -->`;
        case "data-attribute":
          return `<span data-anchor="${id}" style="${opts.style}" class="${opts.className}" ${attributes}></span>`;
        case "hr":
          return `<hr id="${id}" style="${opts.style}" class="${opts.className}" ${attributes}>`;
        default:
          return `<span id="${id}" style="${opts.style}" class="${opts.className}" ${attributes}></span>`;
      }
    };

    return data
      ?.map((content, index) => {
        const anchorTag = generateAnchorTag(index);
        return `${anchorTag}\n\n${content}`;
      })
      .join("\n\n");
  },

  jumpToAnchor: (anchorId: string) => {
    const { mdContents } = get();
    const contentArray = Object.values(mdContents).map(
      (content) => content.content
    );
    let totalLength = 0;
    for (let i = 0; i < contentArray.length; i++) {
      if (anchorId === `md-anchor-${i}`) {
        return totalLength;
      }
      totalLength += contentArray[i].length + 2; // +2 for "\n\n"
    }
    return -1; // Anchor not found
  },

  updateMdContent: async (
    fileKey: string,
    pageNumber: string,
    newContent: string
  ) => {
    try {
      const params: UpdateMarkdownRequest = {
        file_key: fileKey,
        data: {
          [pageNumber]: newContent,
        },
      };

      const result = await updateMarkdownContent(params);

      if (result && result.success) {
        // 更新本地状态
        set((state) => {
          const updatedMdContents = { ...state.mdContents };
          if (updatedMdContents[fileKey]) {
            updatedMdContents[fileKey] = {
              ...updatedMdContents[fileKey],
              content: newContent,
            };
          }

          // 重新计算 allMdContent 和 allMdContentWithAnchor
          const contentArray = Object.values(updatedMdContents).map(
            (content) => content.content
          );
          const newAllMdContent = state.getAllMdContent(contentArray);
          const newAllMdContentWithAnchor =
            state.getContentWithAnchors(contentArray);

          return {
            mdContents: updatedMdContents,
            allMdContent: newAllMdContent,
            allMdContentWithAnchor: newAllMdContentWithAnchor,
          };
        });
      } else {
        throw new Error("Failed to update Markdown content");
      }
    } catch (error) {
      set({ error: error as Error });
      throw error;
    }
  },
}));

export default useMdStore;
