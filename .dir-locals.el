((org-mode
  . ((org-hide-drawer-startup . t)
     (eval . (progn
               (load-file (expand-file-name "tools/ob-fold-hidden.el"
                                            (locate-dominating-file default-directory ".dir-locals.el")))
               (add-hook 'find-file-hook #'el/org-setup-fold-hidden nil t))))))
